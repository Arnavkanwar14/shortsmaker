"""Find a streamer's facecam box in a gameplay stream (for the "split" layout).

Every streamer puts their cam somewhere different, so it is detected per
video: find a face that stays in the same place across sampled frames, then
walk outward from it to the nearest straight edge that is present in almost
every frame (the cam's border: gameplay behind it keeps moving, the border
does not). A side with no such edge runs to the frame edge.
Returns [x, y, w, h] as fractions of the frame, or None if there is no cam.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .util import log

SAMPLES = 16          # frames sampled across the clip
FACE_HITS = 0.3       # the cam face must be found in this share of samples
EDGE_PERSIST = 0.6    # a border pixel is an edge in this share of samples
LINE_COVER = 0.8      # ...along this share of the border's length
MAX_AREA = 0.35       # a "cam" bigger than this is not a cam overlay


def _faces(frame) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h) in pixels. Haar at full resolution: facecam
    faces are small (~100px on 1920), too small for mediapipe's short-range
    model, which downsizes the whole frame to 128px."""
    import cv2
    from .stages.assemble import _cached_model, HAAR_URL
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not xml.exists():
        xml = _cached_model(HAAR_URL, "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [tuple(int(v) for v in f) for f in
            cv2.CascadeClassifier(str(xml)).detectMultiScale(gray, 1.1, 4, minSize=(40, 40))]


def _sample(video: Path, start: float, end: float):
    import cv2
    cap = cv2.VideoCapture(str(video))
    frames = []
    try:
        for i in range(SAMPLES):
            t = start + (end - start) * (i + 0.5) / SAMPLES
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if ok:
                frames.append(f)
    finally:
        cap.release()
    return frames


def _border(edge: np.ndarray, pos: int, stop: int, band: tuple[int, int],
            vertical: bool) -> int:
    """Walk from pos toward stop; return the first line (column if vertical,
    else row) that is a persistent edge along most of `band`, else stop."""
    a, b = band
    step = 1 if stop > pos else -1
    for p in range(pos, stop, step):
        line = edge[a:b, p] if vertical else edge[p, a:b]
        if line.size and (line >= EDGE_PERSIST).mean() >= LINE_COVER:
            return p
    return stop


ROUND_CAM_ASPECT = 1080 / (1920 * 0.40)   # top panel shape a round cam fills
RING_EDGE = 0.4      # share of a round cam's ring that is a persistent edge
RING_STEP = 0.8      # share of the ring where outside changes much more than inside


def _circle(edge: np.ndarray, change: np.ndarray, fcx: float, fcy: float, fw: int):
    """A round cam: the circle around the face where the moving gameplay
    outside meets the still cam background inside, along a persistent edge.
    (Edges alone are ambiguous: the cam's own room is static and full of
    them.) Returns (cx, cy, r, step_share) or None."""
    import cv2
    h, w = edge.shape
    ring_hit = cv2.dilate(edge, np.ones((3, 3), np.uint8)) >= 0.5  # ring may sit 1px off
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    radii = np.arange(int(fw * 1.2), int(fw * 4.5), 2)[:, None]
    best = None
    step = max(2, fw // 15)
    for cy in range(int(fcy - fw), int(fcy + fw) + 1, step):
        for cx in range(int(fcx - fw), int(fcx + fw) + 1, step):
            xo, yo = (cx + (radii + 6) * cos).astype(int), (cy + (radii + 6) * sin).astype(int)
            # a ring running off the frame cannot be a cam border
            ok = ((xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)).all(axis=1)
            if not ok.any():
                continue
            xo, yo = xo.clip(0, w - 1), yo.clip(0, h - 1)
            xi, yi = (cx + (radii - 6) * cos).astype(int).clip(0, w - 1), (cy + (radii - 6) * sin).astype(int).clip(0, h - 1)
            xr, yr = (cx + radii * cos).astype(int).clip(0, w - 1), (cy + radii * sin).astype(int).clip(0, h - 1)
            stepped = (change[yo, xo] > change[yi, xi] * 1.5 + 2).mean(axis=1)
            ringed = ring_hit[yr, xr].mean(axis=1)
            score = np.where(ok & (stepped >= RING_STEP) & (ringed >= RING_EDGE),
                             stepped * ringed, 0)
            i = int(score.argmax())
            if score[i] > 0 and (best is None or score[i] > best[4]):
                best = (cx, cy, int(radii[i, 0]), float(stepped[i]), float(score[i]))
    return best[:4] if best else None


def detect(video: Path, start: float, end: float) -> list[float] | None:
    import cv2
    frames = _sample(video, start, end)
    if not frames:
        return None
    h, w = frames[0].shape[:2]

    # 1. the cam face: the face position that repeats across most samples
    hits = [f for fr in frames for f in _faces(fr)]
    best, best_n = None, 0
    for f in hits:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        same = [g for g in hits if abs(g[0] + g[2] / 2 - cx) < f[2] * 0.5
                and abs(g[1] + g[3] / 2 - cy) < f[3] * 0.5]
        if len(same) > best_n:
            best, best_n = same, len(same)
    if not best or best_n < FACE_HITS * len(frames):
        log.info("facecam: no steady face found (%d/%d samples)", best_n, len(frames))
        return None
    fx, fy, fw, fh = (int(np.median([g[i] for g in best])) for i in range(4))

    # 2. persistent edges: share of samples where each pixel is an edge
    edge = np.zeros((h, w), np.float32)
    for fr in frames:
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (3, 3), 0), 20, 60)
        # thicken by 1px so a border that wobbles a pixel still counts
        edge += cv2.dilate(e, np.ones((3, 3), np.uint8)) > 0
    edge /= len(frames)

    # 3. walk out from the face to the border on each side. The band each
    # walk checks is the face's own span grown a bit, which sits inside the
    # cam even when the border on that axis is still unknown.
    ybd = (max(0, fy - fh // 2), min(h, fy + fh + fh // 2))
    xbd = (max(0, fx - fw // 2), min(w, fx + fw + fw // 2))
    left = _border(edge, fx - fw // 2, 0, ybd, True)
    right = _border(edge, fx + fw + fw // 2, w - 1, ybd, True)
    top = _border(edge, fy - fh // 2, 0, xbd, False)
    bottom = _border(edge, fy + fh + fh // 2, h - 1, xbd, False)

    box = [left / w, top / h, (right - left) / w, (bottom - top) / h]
    if box[2] * box[3] > MAX_AREA or box[2] <= 0 or box[3] <= 0:
        # no straight border: many streamers use a ROUND cam cutout
        gray = np.stack([cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fr in frames]).astype(np.float32)
        change = cv2.GaussianBlur(gray.std(axis=0), (0, 0), 2)
        circle = _circle(edge, change, fx + fw / 2, fy + fh / 2, fw)
        if circle is None:
            log.info("facecam: box %s too big to be an overlay and no round cam; no cam",
                     [round(v, 3) for v in box])
            return None
        ccx, ccy, r, cover = circle
        # largest top-panel-shaped rectangle that fits inside the circle (a
        # little in from the ring, so no gameplay corners show), centred on
        # the head rather than the circle: the face usually sits high in a
        # round cam and a centred box cuts the top of the head off.
        # Solve (A*u)^2 + (|d| + u)^2 = R^2 for the half-height u.
        R, A = 0.94 * r, ROUND_CAM_ASPECT
        d = float(np.clip((fy + fh * 0.5 - ccy) * 0.5, -0.4 * R, 0.4 * R))
        a, b, c = A * A + 1, 2 * abs(d), d * d - R * R
        u = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        rw, rh = 2 * A * u, 2 * u
        box = [float((ccx - rw / 2) / w), float((ccy + d - u) / h), float(rw / w), float(rh / h)]
        log.info("facecam: round cam at (%d,%d) r=%d (outside moves more along %.0f%% of the ring), using %s",
                 ccx, ccy, r, cover * 100, [round(v, 3) for v in box])
        return box
    log.info("facecam: face at (%d,%d) %dx%d, cam box %s", fx, fy, fw, fh,
             [round(v, 3) for v in box])
    return box
