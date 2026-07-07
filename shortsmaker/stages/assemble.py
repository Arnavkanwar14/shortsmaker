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
    """Yields (t_relative_to_clip, frame) about once per second."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        yield round(t - start, 2), frame
        t += 1.0                               # sample 1 frame per second
    cap.release()


def _centers_mediapipe(video: Path, start: float, end: float):
    """mediapipe >= 0.10 Tasks API (legacy mp.solutions was removed)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(
                _cached_model(BLAZEFACE_URL, "blaze_face_short_range.tflite"))),
            min_detection_confidence=0.5))
    samples = []
    for t, frame in _iter_sample_frames(video, start, end):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(img)
        if res.detections:
            # largest face = the shot's subject (active-speaker detection
            # would need audio-visual sync; prominence is a good proxy)
            box = max((d.bounding_box for d in res.detections),
                      key=lambda b: b.width * b.height)
            samples.append((t, (box.origin_x + box.width / 2) / frame.shape[1]))
    return samples


def _centers_haar(video: Path, start: float, end: float):
    """OpenCV Haar cascade fallback (opencv 5 no longer bundles the XMLs)."""
    import cv2
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not xml.exists():
        xml = _cached_model(HAAR_URL, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(str(xml))
    if cascade.empty():
        raise RuntimeError("could not load haar cascade")
    samples = []
    for t, frame in _iter_sample_frames(video, start, end):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])
            samples.append((t, (x + fw / 2) / frame.shape[1]))
    return samples


def face_samples(video: Path, start: float, end: float) -> list[tuple[float, float]]:
    """[(t_rel, normalized center x)] about 1/s, best available detector."""
    for name, fn in (("mediapipe", _centers_mediapipe), ("opencv-haar", _centers_haar)):
        try:
            samples = fn(video, start, end)
        except Exception as e:
            log.info("face detection via %s unavailable (%s)", name, str(e)[:120])
            continue
        if samples:
            return samples
    return []


def build_crop_path(samples: list[tuple[float, float]], threshold: float = 0.08,
                    glide: float = 0.4, confirm: int = 2) -> list[tuple[float, float]]:
    """Hold-then-glide keyframes [(t, cx)] from noisy per-second samples.

    The frame holds still until the subject has clearly moved (> threshold,
    seen on `confirm` consecutive samples -- stops wobble in two-person
    shots), then glides to the new position over `glide` seconds.
    """
    if not samples:
        return []
    cur = samples[0][1]
    kfs = [(0.0, cur)]
    pending: list[tuple[float, float]] = []
    for t, cx in samples[1:]:
        if abs(cx - cur) <= threshold:
            pending = []
            continue
        pending.append((t, cx))
        if len(pending) >= confirm:
            move_t = pending[0][0]
            target = sum(c for _, c in pending) / len(pending)
            glide_start = max(move_t - glide, kfs[-1][0] + 0.05)
            if glide_start < move_t:
                kfs.append((round(glide_start, 2), cur))
            kfs.append((round(move_t, 2), round(target, 4)))
            cur = target
            pending = []
    return kfs


def _crop_x_expr(kfs: list[tuple[float, float]], crop_w: int, w: int) -> str:
    """Piecewise-linear ffmpeg expression for crop x over time.
    Segments use gte*lt (not between) so boundaries never double-count,
    and the result is snapped to even pixels for yuv420."""
    def px(cx: float) -> int:
        return max(0, min(int(cx * w - crop_w / 2), w - crop_w))

    pts = [(t, px(cx)) for t, cx in kfs]
    if len(pts) == 1:
        return str(pts[0][1])
    terms = [f"lt(t,{pts[0][0]})*{pts[0][1]}"]
    for (t0, x0), (t1, x1) in zip(pts, pts[1:]):
        if t1 <= t0:
            continue
        terms.append(f"gte(t,{t0})*lt(t,{t1})*"
                     f"({x0}+({x1}-{x0})*(t-{t0})/({t1}-{t0}))")
    terms.append(f"gte(t,{pts[-1][0]})*{pts[-1][1]}")
    return f"trunc(({'+'.join(terms)})/2)*2"


def crop_filter(cfg: Config, video: Path, clip: dict,
                keeps: list[tuple[float, float]] | None = None) -> str:
    info = ffprobe_video(video)
    w, h = info["width"], info["height"]
    target_ar = cfg.out_width / cfg.out_height          # 9/16

    if w / h <= target_ar + 0.01:                       # already narrow: pad
        return (f"scale={cfg.out_width}:-2,"
                f"pad={cfg.out_width}:{cfg.out_height}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    crop_w = int(h * target_ar) // 2 * 2                # even width
    kfs: list[tuple[float, float]] = []
    if cfg.face_crop:
        samples = face_samples(video, clip["start"], clip["end"])
        kfs = build_crop_path(samples)
        if keeps and kfs:
            # crop time runs on the post-jump-cut timeline
            from ..edits import remap_time
            remapped = []
            for t, cx in kfs:
                rt = remap_time(t, keeps)
                if not remapped or rt > remapped[-1][0]:
                    remapped.append((rt, cx))
            kfs = remapped

    if len(kfs) >= 2:
        x_part = f"x='{_crop_x_expr(kfs, crop_w, w)}':y=0"
        log.info("dynamic speaker crop: %d keyframes", len(kfs))
    else:
        cx = kfs[0][1] if kfs else 0.5
        if kfs:
            log.info("face-aware crop (static): center x = %.2f", cx)
        x = max(0, min(int(cx * w - crop_w / 2), w - crop_w))
        x_part = f"x={x}:y=0"
    return (f"crop=w={crop_w}:h={h}:{x_part},"
            f"scale={cfg.out_width}:{cfg.out_height},setsar=1")


# ----------------------------------------------------------------- main
def run(cfg: Config, video: Path, clip: dict, clip_dir: Path,
        vo_audio: Path | None, caption_words: list[dict],
        keeps: list[tuple[float, float]] | None = None) -> Path:
    """vo_audio None = no voiceover: original audio stays primary and
    caption_words are the source speech instead of TTS words.
    keeps = snappy-cut intervals (relative to clip start) to jump-cut
    dead air/fillers out of the video and original audio."""
    final = clip_dir / "final.mp4"
    if final.exists() and not cfg.force:
        log.info("assemble: final.mp4 exists, skipping")
        return final
    # ffmpeg runs with cwd=clip_dir (for the ass filter); inputs must be absolute
    video = video.resolve()
    if vo_audio is not None:
        vo_audio = vo_audio.resolve()

    ass_file = clip_dir / "captions.ass"
    write_ass(caption_words, ass_file, style=cfg.style,
              preset=cfg.caption_preset, position=cfg.caption_position)

    vf = crop_filter(cfg, video, clip, keeps)
    with_captions = cfg.style != "none" and caption_words
    # ffmpeg is run with cwd=clip_dir so the ass filter gets a plain relative
    # filename -- avoids Windows drive-colon escaping issues in filtergraphs.
    caption_part = f",ass={ass_file.name}" if with_captions else ""

    # original-audio volume: explicit value, else ducked under a voiceover
    # and untouched when the original audio is the only track
    bg_vol = cfg.bg_audio_volume if cfg.bg_audio_volume >= 0 else (
        0.18 if vo_audio is not None else 1.0)

    # jump-cut filters: drop non-keep frames/samples, then regenerate
    # timestamps so the output timeline is continuous
    if keeps:
        from ..edits import select_expr
        expr = select_expr(keeps)
        vcut = f"select='{expr}',setpts=N/FRAME_RATE/TB,"
        acut = f"aselect='{expr}',asetpts=N/SR/TB,"
    else:
        vcut = acut = ""

    def build_args(caption: str) -> list[str]:
        args = ["-ss", str(clip["start"]), "-to", str(clip["end"]),
                "-i", str(video)]
        if vo_audio is not None:
            args += ["-i", str(vo_audio)]
            afilter = (f"[0:a]{acut}volume={bg_vol}[bg];"
                       f"[1:a]volume={cfg.vo_volume},apad[vo];"
                       f"[bg][vo]amix=inputs=2:duration=first:normalize=0[a]")
        else:
            afilter = f"[0:a]{acut}volume={bg_vol}[a]"
        args += [
            "-filter_complex", f"[0:v]{vcut}{vf}{caption}[v];{afilter}",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", final.name,
        ]
        return args

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


def thumbnails(final: Path, clip_dir: Path, n: int = 3) -> list[str]:
    """Candidate cover frames at 25/50/75% -- platforms ask for one at
    upload time. Returns filenames relative to the clip dir."""
    from ..util import media_duration
    dur = media_duration(final)
    names = []
    for i, frac in enumerate((0.25, 0.5, 0.75), 1):
        name = f"thumb_{i}.jpg"
        run_ffmpeg(["-ss", f"{max(dur * frac, 0.1):.2f}", "-i", final.name,
                    "-frames:v", "1", "-q:v", "3", name], cwd=clip_dir)
        names.append(name)
    return names
