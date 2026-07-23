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


TRACK_W = 640          # downscale frames to this width for tracker speed


def _center_biased_seed(frame, sal):
    """Seed box (x,y,w,h in frame px) for the tracker: the salient region
    weighted toward the CENTRE, because the subject is usually central and a
    plain 'largest salient blob' otherwise latches onto bright background off
    to the side. Falls back to a central box when nothing stands out."""
    import cv2
    import numpy as np
    H, W = frame.shape[:2]
    ok, smap = sal.computeSaliency(frame) if sal else (False, None)
    if ok:
        smap = (smap * 255).astype("uint8")
        yy, xx = np.mgrid[0:H, 0:W]
        cw = np.exp(-(((xx - W / 2) / (W * 0.33)) ** 2
                      + ((yy - H / 2) / (H * 0.33)) ** 2))
        weighted = smap.astype(np.float32) * cw
        if weighted.max() > 0:
            norm = (weighted / weighted.max() * 255).astype("uint8")
            _, th = cv2.threshold(norm, 60, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                if w * h > 0.02 * W * H:
                    return (x, y, w, h)
    return (int(W * 0.3), int(H * 0.2), int(W * 0.4), int(H * 0.6))


def detect_subjects(video, start: float, end: float, every: float = 0.2,
                    cuts: list[float] | None = None) -> list[dict]:
    """Track the subject across the clip and return the series auto_track()
    consumes. Uses a CSRT object tracker seeded on the central subject and
    re-seeded at scene cuts -- so the crop actually FOLLOWS the subject as it
    moves left/right instead of the old per-frame 'largest salient blob'
    which had no memory and lurched between the subject and bright background.
    Falls back to per-frame saliency if the tracker isn't available."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    try:
        sal = None
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            pass
        has_csrt = hasattr(cv2, "TrackerCSRT_create")
        if not has_csrt:
            return _detect_subjects_saliency(cap, fps, start, end, sal)

        cut_set = sorted(c - start for c in (cuts or []) if start < c < end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        ok, frame = cap.read()
        if not ok:
            return [{"t": 0.0, "cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0, "conf": 0.0}]
        H, W = frame.shape[:2]
        scale = TRACK_W / W

        def new_tracker(fr):
            seed = _center_biased_seed(fr, sal)
            small = cv2.resize(fr, (TRACK_W, int(H * scale)))
            tk = cv2.TrackerCSRT_create()
            tk.init(small, tuple(int(v * scale) for v in seed))
            return tk

        tracker = new_tracker(frame)
        out, t, next_cut = [], 0.0, 0
        reseed_every = 8.0        # safety re-seed so long-shot drift self-heals
        last_seed = 0.0
        dur = end - start
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + t) * fps))
            ok, frame = cap.read()
            if not ok:
                break
            # re-seed at a scene cut or periodically
            crossed_cut = next_cut < len(cut_set) and t >= cut_set[next_cut]
            if crossed_cut:
                next_cut += 1
            if crossed_cut or (t - last_seed) >= reseed_every:
                tracker = new_tracker(frame)
                last_seed = t
            small = cv2.resize(frame, (TRACK_W, int(H * scale)))
            ok2, box = tracker.update(small)
            if ok2:
                x, y, w, h = [v / scale for v in box]
                out.append({"t": round(t, 2), "cx": round((x + w / 2) / W, 4),
                            "cy": round((y + h / 2) / H, 4),
                            "w": round(w / W, 4), "h": round(h / H, 4),
                            "conf": 0.9 if w / W < 0.95 else 0.2})
            else:
                out.append({"t": round(t, 2), "cx": 0.5, "cy": 0.5,
                            "w": 1.0, "h": 1.0, "conf": 0.0})
                tracker = new_tracker(frame)   # recover
                last_seed = t
            t += every
        return _smooth(out)
    finally:
        cap.release()


def _detect_subjects_saliency(cap, fps, start, end, sal) -> list[dict]:
    """Fallback: per-frame center-of-salient-blob, sampled every 1s."""
    import cv2
    out, t = [], 0.0
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
        t += 1.0
    return _smooth(out)


def _smooth(series: list[dict], win: int = 7) -> list[dict]:
    """Median-smooth cx/cy over a window (wider now that tracking samples
    densely) so residual jitter doesn't reach the crop, then REDUCE the dense
    series to a sparse keyframe set -- keeping a point only where the subject
    has actually moved enough or enough time has passed. Without this a 0.2s
    sample rate would make hundreds of keyframes and a giant ffmpeg
    expression."""
    if len(series) < 3:
        return series
    sm = []
    for i, s in enumerate(series):
        lo, hi = max(0, i - win // 2), min(len(series), i + win // 2 + 1)
        window = series[lo:hi]
        sm.append({**s,
                   "cx": round(median(w["cx"] for w in window), 4),
                   "cy": round(median(w["cy"] for w in window), 4),
                   "conf": median(w["conf"] for w in window)})
    return _reduce(sm)


def _reduce(series: list[dict], move: float = 0.025, max_gap: float = 1.5) -> list[dict]:
    """Keep the first and last sample, plus any where cx/cy moved > `move`
    from the last kept one or > `max_gap` seconds elapsed."""
    if len(series) <= 2:
        return series
    kept = [series[0]]
    for s in series[1:-1]:
        last = kept[-1]
        if (abs(s["cx"] - last["cx"]) > move or abs(s["cy"] - last["cy"]) > move
                or s["t"] - last["t"] > max_gap):
            kept.append(s)
    kept.append(series[-1])
    return kept

# Hard ceiling on any auto punch-in. Deliberately low: over-zooming crops
# the subject (only its midsection ends up in frame) and softens the image,
# both reported as worse than no zoom. The default behaviour is PAN, not
# zoom -- on a landscape source the 9:16 slice is already a big crop, so we
# just slide it to follow the subject at zoom ~1.0.
MAX_ZOOM = 1.25
# below this the punch-in isn't worth it -- snap back to 1.0 (pan only)
MIN_USEFUL_ZOOM = 1.10


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def auto_track(subjects: list[dict], clip_dur: float,
               src_w: int = 1920, src_h: int = 1080,
               out_ar: float = 9 / 16) -> list[dict]:
    """Build a default reframe track from a time series of subject boxes.

    subjects: [{"t", "cx", "cy", "w", "h", "conf"}] normalized (0..1).

    PAN-FIRST and conservative: the 9:16 crop is slid to keep the subject
    centred at zoom 1.0. A gentle punch-in is applied ONLY when the subject
    is genuinely NARROWER than that crop already is (so zooming can actually
    frame it better without cutting it off), and even then it's capped hard
    at MAX_ZOOM. A subject wider than the crop, or no confident subject,
    stays at zoom 1.0 -- we never zoom into a subject we'd only clip.
    """
    if not subjects:
        return [{"t": 0.0, "zoom": 1.0, "cx": 0.5, "cy": 0.5}]

    # PAN-ONLY: auto lock-on never zooms (zoom stays 1.0) -- over-zooming
    # cropped the subject in real runs and the user asked, emphatically, for
    # no auto-zoom on any clip. On a landscape source zoom-1 is already the
    # 9:16 slice, so this just slides that slice to follow the subject; on an
    # already-vertical source it's a no-op. Zoom is still available, but only
    # when the user adds it themselves in the editor.
    kfs = []
    for s in subjects:
        kfs.append({"t": s["t"], "zoom": 1.0,
                    "cx": _clamp(s.get("cx", 0.5), 0.0, 1.0),
                    "cy": _clamp(s.get("cy", 0.5), 0.0, 1.0)})

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


def crop_chain(track: list[dict], iw: int, ih: int, out_w: int, out_h: int,
               remap=None) -> str | None:
    """The crop+scale filter body (no leading watermark trim, no trailing
    caption/[v] label) for use INSIDE assemble's existing filtergraph, so it
    inherits watermark-trim and jump-cut handling. `remap` optionally maps a
    keyframe's clip-time onto the post-jump-cut timeline (assemble passes
    edits.remap_time bound to its keeps). Returns None when the track is a
    no-op on an already-vertical source (caller keeps its own framing)."""
    zoom = clip_zoom(track)
    target_ar = out_w / out_h
    if iw / ih >= target_ar:
        base_w, base_h = ih * target_ar, ih
    else:
        base_w, base_h = iw, iw / target_ar
    crop_w = int(base_w / zoom) // 2 * 2
    crop_h = int(base_h / zoom) // 2 * 2

    source_is_vertical = abs(iw / ih - target_ar) < 0.02
    static_center = all(abs(k["cx"] - 0.5) < 0.02 and abs(k["cy"] - 0.5) < 0.02
                        for k in track)
    if source_is_vertical and zoom <= MIN_USEFUL_ZOOM and static_center:
        return None

    tr = track
    if remap is not None:
        tr, seen = [], set()
        for k in track:
            rt = round(remap(k["t"]), 3)
            if rt not in seen:      # keep first at each remapped instant
                seen.add(rt)
                tr.append({**k, "t": rt})
    x_expr = crop_pan_expr(tr, iw, ih, zoom, crop_w, crop_h, "x")
    y_expr = crop_pan_expr(tr, iw, ih, zoom, crop_w, crop_h, "y")
    return (f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}',"
            f"scale={out_w}:{out_h},setsar=1")


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
