"""Subject-following reframe track: a per-clip, time-keyframed zoom+pan that
recentres the shot onto the subject -- and, crucially, WORKS ON ALREADY-
VERTICAL sources (which the old face-crop path skipped entirely, so the
"auto lock-on" did nothing on them).

A track is a list of keyframes, each clip-relative:
    {"t": seconds, "zoom": >=1.0, "cx": 0..1, "cy": 0..1}
zoom is how far to punch in (1.0 = whole frame, no reframe); cx/cy is the
normalized centre of the kept region. The track is stored per clip as
reframe.json so it can be auto-generated, reviewed, and hand-edited, then
re-rendered without re-running the whole pipeline.

v1 render: a fixed zoom per clip (the track's median zoom) + fully
time-varying pan via ffmpeg `crop` time-expressions (verified to render
cleanly and at full quality). Per-keyframe zoom-over-time is a later layer
on the same model; the schema already carries per-keyframe zoom so nothing
has to change here when it lands.
"""
from __future__ import annotations

import logging
from statistics import median

log = logging.getLogger("shortsmaker")

_SAL = None    # cached saliency detector | False if unavailable


def _salient_box(frame):
    """(cx, cy, w, h, conf) of the dominant salient region, normalized, or
    None. conf is low when the salient area fills the frame (establishing
    shot / full-frame explosion -- no useful subject to punch into) or is
    tiny noise."""
    global _SAL
    import cv2
    import numpy as np
    if _SAL is None:
        try:
            _SAL = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            _SAL = False
    if not _SAL:
        return None
    ok, smap = _SAL.computeSaliency(frame)
    if not ok:
        return None
    smap = (smap * 255).astype("uint8")
    _, th = cv2.threshold(smap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = frame.shape[:2]
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    area_frac = (bw * bh) / (W * H)
    # confident only when the blob is a real subject: not the whole frame,
    # not a speck, and reasonably compact (its bbox mostly filled)
    fill = cv2.contourArea(c) / max(bw * bh, 1)
    conf = 0.0
    if 0.03 < area_frac < 0.7 and fill > 0.35:
        conf = min(1.0, fill)
    return ((x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H, conf)


def detect_subjects(video, start: float, end: float,
                    every: float = 1.0) -> list[dict]:
    """Sample the clip ~every seconds and locate the subject per sample.
    Returns the series auto_track() consumes. Cheap: saliency only, no model
    download, works on any content (incl. non-human subjects the old face
    detector missed)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = []
    try:
        t = 0.0
        while t < end - start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + t) * fps))
            ok, frame = cap.read()
            if ok:
                box = _salient_box(frame)
                if box:
                    cx, cy, w, h, conf = box
                    out.append({"t": round(t, 2), "cx": cx, "cy": cy,
                                "w": w, "h": h, "conf": conf})
                else:
                    out.append({"t": round(t, 2), "cx": 0.5, "cy": 0.5,
                                "w": 1.0, "h": 1.0, "conf": 0.0})
            t += every
    finally:
        cap.release()
    return _smooth(out)


def _smooth(series: list[dict], win: int = 3) -> list[dict]:
    """Median-smooth cx/cy and drop lone confidence spikes so one bad
    frame can't yank the crop (the lurch bug the old path had)."""
    if len(series) < 3:
        return series
    out = []
    for i, s in enumerate(series):
        lo, hi = max(0, i - win // 2), min(len(series), i + win // 2 + 1)
        window = series[lo:hi]
        out.append({**s,
                    "cx": median(w["cx"] for w in window),
                    "cy": median(w["cy"] for w in window),
                    "conf": median(w["conf"] for w in window)})
    return out

# never punch in tighter than this (upscaling a vertical source softens it)
MAX_ZOOM = 1.6
# below this the punch-in isn't worth the quality hit -- snap back to 1.0
MIN_USEFUL_ZOOM = 1.08


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def auto_track(subjects: list[dict], clip_dur: float,
               out_ar: float = 9 / 16) -> list[dict]:
    """Build a default reframe track from a time series of subject boxes.

    subjects: [{"t", "cx", "cy", "w", "h", "conf"}] normalized (0..1), where
    w/h are the subject's fractional extent and conf 0..1 its detection
    confidence (0 = nothing found -> that sample contributes no punch-in).

    Conservative by design: only punches in when there's a confident,
    reasonably-sized, off-centre subject; an establishing shot with nothing
    found stays at zoom 1.0 (full frame) so we never zoom into empty space.
    """
    if not subjects:
        return [{"t": 0.0, "zoom": 1.0, "cx": 0.5, "cy": 0.5}]

    kfs = []
    for s in subjects:
        conf = s.get("conf", 0.0)
        if conf < 0.35 or s.get("w", 1.0) >= 0.95:
            # nothing solid, or subject already fills the width -> no punch-in
            kfs.append({"t": s["t"], "zoom": 1.0,
                        "cx": _clamp(s.get("cx", 0.5), 0.0, 1.0),
                        "cy": _clamp(s.get("cy", 0.5), 0.0, 1.0)})
            continue
        # zoom just enough to make the subject fill ~80% of the frame width,
        # capped, and never so much that the kept region can't hold it
        want = _clamp(0.8 / max(s["w"], 0.2), 1.0, MAX_ZOOM)
        if want < MIN_USEFUL_ZOOM:
            want = 1.0
        kfs.append({"t": s["t"], "zoom": round(want, 3),
                    "cx": _clamp(s["cx"], 0.0, 1.0),
                    "cy": _clamp(s["cy"], 0.0, 1.0)})

    kfs.sort(key=lambda k: k["t"])
    if kfs[0]["t"] > 0.01:
        kfs.insert(0, dict(kfs[0], t=0.0))
    return kfs


def clip_zoom(track: list[dict]) -> float:
    """The single zoom v1 renders at: the median of the track's non-trivial
    zooms (ignores 1.0 'no punch-in' samples so a few full-frame beats don't
    drag a mostly-punched-in clip back to no zoom). 1.0 if the whole clip is
    full-frame."""
    zs = [k["zoom"] for k in track if k["zoom"] > MIN_USEFUL_ZOOM]
    return round(median(zs), 3) if zs else 1.0


def crop_pan_expr(track: list[dict], iw: int, ih: int, zoom: float,
                  crop_w: int, crop_h: int, axis: str) -> str:
    """Piecewise-linear ffmpeg expression for the crop x (or y) origin over
    time, following the track's cx/cy at the fixed clip `zoom`. Glides
    between keyframes; clamps so the kept region never leaves the frame."""
    span = iw if axis == "x" else ih
    crop = crop_w if axis == "x" else crop_h
    key = "cx" if axis == "x" else "cy"

    def origin(k: dict) -> int:
        c = k[key]
        return int(_clamp(c * span - crop / 2, 0, span - crop))

    pts = [(k["t"], origin(k)) for k in track]
    # collapse consecutive identical positions to keep the expression small
    dedup = [pts[0]]
    for t, x in pts[1:]:
        if x != dedup[-1][1]:
            dedup.append((t, x))
    pts = dedup
    if len(pts) == 1:
        return str(pts[0][1])

    terms = [f"lt(t,{pts[0][0]:.2f})*{pts[0][1]}"]
    for (t0, x0), (t1, x1) in zip(pts, pts[1:]):
        if t1 <= t0:
            continue
        cond = f"gte(t,{t0:.2f})*lt(t,{t1:.2f})"
        terms.append(f"{cond}*({x0}+({x1}-{x0})*(t-{t0:.2f})/({t1 - t0:.2f}))")
    terms.append(f"gte(t,{pts[-1][0]:.2f})*{pts[-1][1]}")
    return f"trunc(({'+'.join(terms)})/2)*2"


def render_filter(track: list[dict], iw: int, ih: int,
                  out_w: int, out_h: int) -> str:
    """ffmpeg -vf chain for the reframe track: crop the fixed-zoom 9:16
    region, panning over time, then scale to output. Returns '' when the
    track is a no-op (zoom 1.0 and centred) so the caller can skip it."""
    zoom = clip_zoom(track)
    target_ar = out_w / out_h
    # base 9:16 region that fits the source, then divide by zoom
    if iw / ih >= target_ar:
        base_w, base_h = ih * target_ar, ih
    else:
        base_w, base_h = iw, iw / target_ar
    crop_w = int(base_w / zoom) // 2 * 2
    crop_h = int(base_h / zoom) // 2 * 2

    # A no-op is ONLY valid when the source is ALREADY ~9:16 (the base region
    # covers essentially the whole frame): then zoom-1 + centred means there's
    # genuinely nothing to crop. On a LANDSCAPE source the base region is a
    # narrow vertical slice, so cropping is mandatory even when the subject
    # sits centred -- returning "" there would leave the output landscape.
    source_is_vertical = abs(iw / ih - target_ar) < 0.02
    static_center = all(abs(k["cx"] - 0.5) < 0.02 and abs(k["cy"] - 0.5) < 0.02
                        for k in track)
    if source_is_vertical and zoom <= MIN_USEFUL_ZOOM and static_center:
        return ""   # nothing to do -- caller keeps its existing framing

    x_expr = crop_pan_expr(track, iw, ih, zoom, crop_w, crop_h, "x")
    y_expr = crop_pan_expr(track, iw, ih, zoom, crop_w, crop_h, "y")
    xp = x_expr if not x_expr.lstrip("-").isdigit() else x_expr
    yp = y_expr if not y_expr.lstrip("-").isdigit() else y_expr
    return (f"crop=w={crop_w}:h={crop_h}:x='{xp}':y='{yp}',"
            f"scale={out_w}:{out_h},setsar=1")
