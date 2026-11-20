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
        log.info("facecam: box %s too big to be an overlay; no cam", [round(v, 3) for v in box])
        return None
    log.info("facecam: face at (%d,%d) %dx%d, cam box %s", fx, fy, fw, fh,
             [round(v, 3) for v in box])
    return box
