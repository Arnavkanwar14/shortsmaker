"""STAGE 6 -- ASSEMBLE: 9:16 crop, audio duck+mix, kinetic captions, export.

Face-aware crop: mediapipe face detection over sampled frames inside the
clip window; median face center steers the crop. Falls back to center
crop if mediapipe is missing or no faces found.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..captions import write_ass
from ..config import Config
from ..util import ffprobe_video, run_ffmpeg

log = logging.getLogger("shortsmaker")


# ------------------------------------------------------------ face crop
BLAZEFACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
HAAR_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/data/"
            "haarcascades/haarcascade_frontalface_default.xml")


def _cached_model(url: str, filename: str) -> Path:
    """Tiny one-time model download cached under ~/.cache/shortsmaker."""
    cache = Path.home() / ".cache" / "shortsmaker"
    model = cache / filename
    if not model.exists():
        import requests
        cache.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s (one-time) ...", filename)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        model.write_bytes(r.content)
    return model


def _iter_sample_frames(video: Path, start: float, end: float):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        t += 1.0                               # sample 1 frame per second
    cap.release()


def _centers_mediapipe(video: Path, start: float, end: float) -> list[float]:
    """mediapipe >= 0.10 Tasks API (legacy mp.solutions was removed)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(
                _cached_model(BLAZEFACE_URL, "blaze_face_short_range.tflite"))),
            min_detection_confidence=0.5))
    centers = []
    for frame in _iter_sample_frames(video, start, end):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(img)
        if res.detections:
            box = res.detections[0].bounding_box
            centers.append((box.origin_x + box.width / 2) / frame.shape[1])
    return centers


def _centers_haar(video: Path, start: float, end: float) -> list[float]:
    """OpenCV Haar cascade fallback (opencv 5 no longer bundles the XMLs)."""
    import cv2
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not xml.exists():
        xml = _cached_model(HAAR_URL, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(str(xml))
    if cascade.empty():
        raise RuntimeError("could not load haar cascade")
    centers = []
    for frame in _iter_sample_frames(video, start, end):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])
            centers.append((x + fw / 2) / frame.shape[1])
    return centers


def face_center_x(video: Path, start: float, end: float) -> float | None:
    """Median normalized (0..1) face-center x inside the window, or None."""
    for name, fn in (("mediapipe", _centers_mediapipe), ("opencv-haar", _centers_haar)):
        try:
            centers = fn(video, start, end)
        except Exception as e:
            log.info("face detection via %s unavailable (%s)", name, str(e)[:120])
            continue
        if centers:
            centers.sort()
            return centers[len(centers) // 2]
    return None


def crop_filter(cfg: Config, video: Path, clip: dict) -> str:
    info = ffprobe_video(video)
    w, h = info["width"], info["height"]
    target_ar = cfg.out_width / cfg.out_height          # 9/16

    if w / h <= target_ar + 0.01:                       # already narrow: pad
        return (f"scale={cfg.out_width}:-2,"
                f"pad={cfg.out_width}:{cfg.out_height}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    crop_w = int(h * target_ar) // 2 * 2                # even width
    cx = 0.5
    if cfg.face_crop:
        found = face_center_x(video, clip["start"], clip["end"])
        if found is not None:
            cx = found
            log.info("face-aware crop: center x = %.2f", cx)
    x = max(0, min(int(cx * w - crop_w / 2), w - crop_w))
    return (f"crop={crop_w}:{h}:{x}:0,"
            f"scale={cfg.out_width}:{cfg.out_height},setsar=1")


# ----------------------------------------------------------------- main
def run(cfg: Config, video: Path, clip: dict, clip_dir: Path,
        vo_audio: Path, vo_words: list[dict]) -> Path:
    final = clip_dir / "final.mp4"
    if final.exists() and not cfg.force:
        log.info("assemble: final.mp4 exists, skipping")
        return final
    # ffmpeg runs with cwd=clip_dir (for the ass filter); inputs must be absolute
    video = video.resolve()
    vo_audio = vo_audio.resolve()

    ass_file = clip_dir / "captions.ass"
    write_ass(vo_words, ass_file, style=cfg.style)

    vf = crop_filter(cfg, video, clip)
    with_captions = cfg.style != "none"
    # ffmpeg is run with cwd=clip_dir so the ass filter gets a plain relative
    # filename -- avoids Windows drive-colon escaping issues in filtergraphs.
    caption_part = f",ass={ass_file.name}" if with_captions else ""

    def build_args(caption: str) -> list[str]:
        return [
            "-ss", str(clip["start"]), "-to", str(clip["end"]),
            "-i", str(video), "-i", str(vo_audio),
            "-filter_complex",
            f"[0:v]{vf}{caption}[v];"
            f"[0:a]volume={cfg.bg_audio_volume}[bg];"
            f"[1:a]apad[vo];"
            f"[bg][vo]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", final.name,
        ]

    try:
        run_ffmpeg(build_args(caption_part), cwd=clip_dir)
    except RuntimeError as e:
        if with_captions:
            # some static ffmpeg builds lack libass -- deliver uncaptioned
            # rather than failing the clip, and say so loudly
            log.warning("caption burn failed (ffmpeg without libass?); "
                        "retrying without captions. %s", str(e)[:300])
            run_ffmpeg(build_args(""), cwd=clip_dir)
        else:
            raise

    log.info("exported %s", final)
    return final
