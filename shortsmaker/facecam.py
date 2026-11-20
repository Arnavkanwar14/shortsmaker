"""Find a streamer's facecam box in a gameplay stream (for the "split" layout).

SHARED FILE: an identical copy lives in supoclip/backend/src/media/facecam.py.
Keep the two in sync (copy the whole file), so both tools detect cams the
same way.

Every streamer's cam is different (corner, edge, floating, circle, with or
without a border), so nothing about position or shape is assumed:
  1. Faces that stay in one place across the clip are cam candidates. Haar
     also fires on gameplay (grass, NPCs) and on a still shot those false
     hits repeat in place, so up to 8 candidates are tried, most frequent
     first, and each must prove a real cam border around it.
  2. A cam border is where the gameplay outside keeps changing while the cam
     background inside stays still, usually along a persistent edge. Edges
     alone are not enough: the cam's own room (lamp, shelves, mic) is static
     and full of them.
  3. Round cams are tried first (their check is strict), then rectangles,
     walking out from the face on each side. A side with no border runs to
     the frame edge (cams that touch the frame edge).
Returns (x, y, w, h) in source pixels, or None (callers fall back to a
non-split layout; a cam is never guessed).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SAMPLES = 16
MIN_FACE_HITS = 2       # a small cam face is often seen in only a few samples
MAX_CANDIDATES = 8
MAX_AREA = 0.35         # a "cam" bigger than this share of the frame is not an inset
MIN_AREA = 0.004
EDGE_PERSIST = 0.5      # a border pixel is an edge in this share of samples
# rectangle sides: share of the side that must be an edge / a motion step
SIDE_EDGE, SIDE_STEP = 0.6, 0.5
# round cams: share of the ring that must be an edge / a motion step
RING_EDGE, RING_STEP = 0.4, 0.8
# the shape a round cam is cropped to (the split layout's top panel at 40%)
ROUND_CAM_ASPECT = 1080 / (1920 * 0.40)


def sample_frames(path: Path, start: float, end: float, width: int, height: int,
                  ffmpeg: str = "ffmpeg") -> list:
    """BGR frames spread across [start, end], decoded by ffmpeg (opencv
    builds often cannot decode AV1, which YouTube commonly serves)."""
    frames = []
    for i in range(SAMPLES):
        t = start + (end - start) * (i + 0.5) / SAMPLES
        raw = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, timeout=60).stdout
        if len(raw) == width * height * 3:
            frames.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3))
    return frames


def _stepped(change, out_ys, out_xs, in_ys, in_xs):
    """Share of points where the outside changes clearly more than inside."""
    return (change[out_ys, out_xs] > change[in_ys, in_xs] * 1.5 + 2).mean(axis=-1)


def _circle(edge, change, fcx, fcy, fw):
    """Round cam around the face: (cx, cy, r) or None."""
    h, w = edge.shape
    ring_hit = cv2.dilate(edge, np.ones((3, 3), np.uint8)) >= EDGE_PERSIST
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    radii = np.arange(int(fw * 1.2), int(fw * 4.5), 2)[:, None]
    best = None
    step = max(2, fw // 15)
    for cy in range(int(fcy - fw), int(fcy + fw) + 1, step):
        for cx in range(int(fcx - fw), int(fcx + fw) + 1, step):
            xo, yo = (cx + (radii + 6) * cos).astype(int), (cy + (radii + 6) * sin).astype(int)
            ok = ((xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)).all(axis=1)
            if not ok.any():
                continue
            xi = (cx + (radii - 6) * cos).astype(int).clip(0, w - 1)
            yi = (cy + (radii - 6) * sin).astype(int).clip(0, h - 1)
            xr = (cx + radii * cos).astype(int).clip(0, w - 1)
            yr = (cy + radii * sin).astype(int).clip(0, h - 1)
            stepped = _stepped(change, yo.clip(0, h - 1), xo.clip(0, w - 1), yi, xi)
            ringed = ring_hit[yr, xr].mean(axis=1)
            score = np.where(ok & (stepped >= RING_STEP) & (ringed >= RING_EDGE),
                             stepped * ringed, 0)
            i = int(score.argmax())
            if score[i] > 0 and (best is None or score[i] > best[3]):
                best = (cx, cy, int(radii[i, 0]), float(score[i]))
    return best[:3] if best else None


def _side(edge, change, pos, stop, band, vertical):
    """Walk from pos toward stop for the cam border on one side: a line that
    is a persistent edge AND has more change outside than inside along most
    of `band`. None if the walk reaches `stop` (the frame edge)."""
    a, b = band
    h, w = edge.shape
    lim = (w if vertical else h) - 1
    d = 1 if stop > pos else -1
    idx = np.arange(a, b)
    for p in range(pos, stop, d):
        po, pi = min(max(p + 6 * d, 0), lim), min(max(p - 6 * d, 0), lim)
        if vertical:
            line = edge[a:b, p]
            st = _stepped(change, idx, np.full_like(idx, po), idx, np.full_like(idx, pi))
        else:
            line = edge[p, a:b]
            st = _stepped(change, np.full_like(idx, po), idx, np.full_like(idx, pi), idx)
        if (line >= EDGE_PERSIST).mean() >= SIDE_EDGE and st >= SIDE_STEP:
            return p
    return None


def _rect(edge, change, fx, fy, fw, fh):
    h, w = edge.shape
    # walk along the face's own span: it is inside the cam on both axes
    ys, xs = (fy, fy + fh), (fx, fx + fw)
    left = _side(edge, change, fx, 0, ys, True)
    right = _side(edge, change, fx + fw, w - 1, ys, True)
    top = _side(edge, change, fy, 0, xs, False)
    bottom = _side(edge, change, fy + fh, h - 1, xs, False)
    found = sum(v is not None for v in (left, right, top, bottom))
    left = 0 if left is None else left
    right = w if right is None else right
    top = 0 if top is None else top
    bottom = h if bottom is None else bottom
    # a floating cam shows all 4 borders, a corner cam 2, an edge cam 3;
    # fewer than 2 means the walk did not really find a box
    if found < 2:
        return None
    return left, top, right - left, bottom - top


def _round_to_box(circle, fy, fh):
    """Largest top-panel-shaped rectangle inside the circle (a little in from
    the ring, so no gameplay corners show), nudged toward the head: solve
    (A*u)^2 + (|d| + u)^2 = R^2 for the half-height u."""
    ccx, ccy, r = circle
    R, A = 0.95 * r, ROUND_CAM_ASPECT
    d = float(np.clip((fy + fh * 0.5 - ccy) * 0.5, -0.4 * R, 0.4 * R))
    a, b, c = A * A + 1, 2 * abs(d), d * d - R * R
    u = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return int(ccx - A * u), int(ccy + d - u), int(2 * A * u), int(2 * u)


def _faces(frames, cascade_path):
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return []
    w = frames[0].shape[1]
    size = max(32, w // 60)
    return [tuple(int(v) for v in f) for fr in frames for f in
            cascade.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.1, 4,
                                     minSize=(size, size))]


def find_cam(frames: list, cascade_path) -> tuple | None:
    """(x, y, w, h) of the facecam in pixels, or None. See module docstring."""
    if len(frames) < 8:
        logger.info("facecam: only %d frames decoded; no cam", len(frames))
        return None
    h, w = frames[0].shape[:2]
    hits = _faces(frames, cascade_path)

    def near(f, g):
        return (abs(g[0] + g[2] / 2 - f[0] - f[2] / 2) < f[2] * .5
                and abs(g[1] + g[3] / 2 - f[1] - f[3] / 2) < f[3] * .5)
    groups = []
    for f in sorted(hits, key=lambda f: -sum(near(f, g) for g in hits)):
        if any(near(f, g) for g in (c[0] for c in groups)):
            continue
        same = [g for g in hits if near(f, g)]
        if len(same) >= MIN_FACE_HITS:
            groups.append((f, same))
    if not groups:
        logger.info("facecam: no steady face (%d face hits)", len(hits))
        return None

    gray = np.stack([cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fr in frames])
    edge = np.zeros((h, w), np.float32)
    for g in gray:
        e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 20, 60)
        edge += cv2.dilate(e, np.ones((3, 3), np.uint8)) > 0
    edge /= len(frames)
    change = cv2.GaussianBlur(gray.astype(np.float32).std(axis=0), (0, 0), 2)

    for _, same in groups[:MAX_CANDIDATES]:
        fx, fy, fw, fh = (int(np.median([g[i] for g in same])) for i in range(4))
        if fw * fh > 0.04 * w * h:          # a big face is the content, not an inset
            continue
        circle = _circle(edge, change, fx + fw / 2, fy + fh / 2, fw)
        if circle:
            box, kind = _round_to_box(circle, fy, fh), "round"
        else:
            box, kind = _rect(edge, change, fx, fy, fw, fh), "rect"
        if not box:
            continue
        x, y, bw, bh = box
        if not (MIN_AREA * w * h <= bw * bh <= MAX_AREA * w * h) or not 0.4 <= bw / bh <= 3:
            continue
        if kind == "rect":
            # the picture outside the box must change much more over time:
            # rules out a static shot (podcast, desk) that has edges
            inside = np.zeros((h, w), bool)
            inside[y:y + bh, x:x + bw] = True
            if change[~inside].mean() < 1.5 * max(change[inside].mean(), 1.0):
                continue
        logger.info("facecam: %s cam %s around face at (%d,%d) %dx%d", kind,
                    (x, y, bw, bh), fx, fy, fw, fh)
        return x, y, bw, bh
    logger.info("facecam: %d face candidates, none inside a cam border", len(groups))
    return None


def detect(video: Path, start: float, end: float) -> list[float] | None:
    """shortsmaker entry point: the cam box as [x, y, w, h] frame fractions."""
    from .stages.assemble import HAAR_URL, _cached_model
    from .util import ffmpeg_exe, ffprobe_video
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not xml.exists():
        xml = _cached_model(HAAR_URL, "haarcascade_frontalface_default.xml")
    info = ffprobe_video(video)
    w, h = info["width"], info["height"]
    box = find_cam(sample_frames(video, start, end, w, h, ffmpeg_exe()), xml)
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h] if box else None
