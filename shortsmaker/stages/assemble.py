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


# --- detectors are cached at module level: model load is the slow part,
# --- and it used to be paid once per CLIP instead of once per process
_MP = None       # (mediapipe module, detector) | False when unavailable
_HAAR = None     # cv2.CascadeClassifier | False when unavailable


def _detect_faces(frame) -> list[float]:
    """All normalized face-center x positions (0..1) in one frame, largest
    first. mediapipe first, haar fallback."""
    global _MP, _HAAR
    import cv2
    if _MP is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            det = vision.FaceDetector.create_from_options(
                vision.FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(
                        _cached_model(BLAZEFACE_URL, "blaze_face_short_range.tflite"))),
                    min_detection_confidence=0.5))
            _MP = (mp, det)
        except Exception as e:
            log.info("mediapipe unavailable (%s); using haar cascade", str(e)[:100])
            _MP = False
    if _MP:
        mp, det = _MP
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = det.detect(img)
        boxes = sorted(res.detections,
                       key=lambda d: -(d.bounding_box.width * d.bounding_box.height))
        return [(b.bounding_box.origin_x + b.bounding_box.width / 2) / frame.shape[1]
                for b in boxes]
    if _HAAR is None:
        try:
            xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if not xml.exists():
                xml = _cached_model(HAAR_URL, "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(str(xml))
            _HAAR = cascade if not cascade.empty() else False
        except Exception:
            _HAAR = False
    if _HAAR:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        faces = sorted(faces, key=lambda f: -(f[2] * f[3]))
        return [(x + fw / 2) / frame.shape[1] for x, _, fw, _ in faces]
    return []


def choose_center(faces: list[float], prev: float) -> float | None:
    """Pick which face to frame on. A single face is unambiguous; with
    several, prefer whichever is closest to the PREVIOUS chosen position
    (temporal continuity) instead of always the largest -- in a two-person
    shot, "largest" flips between people as their relative size shifts,
    which made the crop settle on the average of both instead of either."""
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]
    return min(faces, key=lambda f: abs(f - prev))


def shot_bounds(cuts: list[float], start: float, end: float,
                min_shot: float = 0.8) -> list[tuple[float, float]]:
    """Clip-relative shot segments [(s0, s1)] from absolute scene cuts;
    fragments shorter than min_shot merge into the previous shot."""
    dur = round(end - start, 2)
    bounds = [0.0]
    for c in sorted(cuts):
        t = round(c - start, 2)
        if min_shot <= t <= dur - min_shot and t - bounds[-1] >= min_shot:
            bounds.append(t)
    bounds.append(dur)
    return list(zip(bounds, bounds[1:]))


SUBSHOT_SPAN = 2.0   # re-detect at least this often WITHIN a shot, so a pan
                     # or a subject walking is tracked even without a cut


def sub_windows(s0: float, s1: float, span: float = SUBSHOT_SPAN) -> list[tuple[float, float]]:
    """Subdivide a shot into <=span-second windows for re-detection."""
    out, t = [], s0
    while t < s1:
        t_end = min(t + span, s1)
        out.append((round(t, 2), round(t_end, 2)))
        t = t_end
    return out


def _sample_fracs(span: float) -> tuple[float, ...]:
    if span < 2:
        return (0.5,)
    if span < 4:
        return (0.3, 0.7)
    return (0.2, 0.5, 0.8)


def shot_crop_keyframes(video: Path, start: float, end: float,
                        cuts: list[float]) -> list[tuple[float, float, bool]]:
    """Crop position keyframes [(t, cx, is_cut)]: is_cut marks a keyframe
    that starts a new SHOT (a real scene cut) vs one that's a re-detection
    within the same continuous take. Re-detects every SUBSHOT_SPAN seconds
    within a shot so movement inside a take is tracked instead of frozen
    at the shot's start position; is_cut lets the caller snap instantly at
    real cuts but glide smoothly between re-detections in the same shot."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    kfs: list[tuple[float, float, bool]] = []
    prev = 0.5
    try:
        for s0, s1 in shot_bounds(cuts, start, end):
            first_in_shot = True
            for sub0, sub1 in sub_windows(s0, s1):
                span = sub1 - sub0
                centers = []
                for f in _sample_fracs(span):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + sub0 + span * f) * fps))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    faces = _detect_faces(frame)
                    cx = choose_center(faces, prev)
                    if cx is not None:
                        centers.append(cx)
                cx = sorted(centers)[len(centers) // 2] if centers else prev
                # ignore sub-5% shifts: re-framing for nothing looks like jitter
                if not kfs or abs(cx - kfs[-1][1]) >= 0.05:
                    kfs.append((sub0, round(cx, 4), first_in_shot))
                    first_in_shot = False
                prev = cx     # update every window: prev drives face
                              # disambiguation, not just the jitter gate
    finally:
        cap.release()
    return kfs


def _crop_x_mixed_expr(kfs: list[tuple[float, float, bool]], crop_w: int, w: int) -> str:
    """Piecewise ffmpeg x expression: HARD SNAP at real scene cuts (gliding
    across a genuine edit reads as a pan glitch) but smooth GLIDE between
    re-detections within the same continuous shot (natural camera-follow
    instead of an instant jump mid-take, which could leave the subject
    out of frame for the remainder of a long take). gte/lt bounds never
    double-count; result snapped to even pixels."""
    def px(cx: float) -> int:
        return max(0, min(int(cx * w - crop_w / 2), w - crop_w))

    pts = [(t, px(cx), is_cut) for t, cx, is_cut in kfs]
    if len(pts) == 1:
        return str(pts[0][1])

    terms = [f"lt(t,{pts[0][0]})*{pts[0][1]}"]
    for i in range(len(pts) - 1):
        t0, x0, _ = pts[i]
        t1, x1, is_cut1 = pts[i + 1]
        if t1 <= t0:
            continue
        cond = f"gte(t,{t0})*lt(t,{t1})"
        if is_cut1:
            terms.append(f"{cond}*{x0}")          # hold, then hard-snap at the cut
        else:
            terms.append(f"{cond}*({x0}+({x1}-{x0})*(t-{t0})/({t1}-{t0}))")  # glide
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
    kfs: list[tuple[float, float, bool]] = []
    if cfg.face_crop:
        from ..util import read_json
        scenes_file = cfg.run_dir / "scenes.json"
        cuts = read_json(scenes_file) if scenes_file.exists() else []
        try:
            kfs = shot_crop_keyframes(video, clip["start"], clip["end"], cuts)
        except Exception as e:
            log.info("face crop failed (%s); using center crop", str(e)[:120])
        if keeps and kfs:
            # crop time runs on the post-jump-cut timeline
            from ..edits import remap_time
            remapped = []
            for t, cx, is_cut in kfs:
                rt = remap_time(t, keeps)
                if not remapped or rt > remapped[-1][0]:
                    remapped.append((rt, cx, is_cut))
            kfs = remapped

    # "balanced" crops a wider (less zoomed) slice and fills the remaining
    # top/bottom sliver with a blurred copy of the same frame, instead of
    # blowing the exact-fill crop up to fill the whole 1080x1920 frame --
    # "tight" is a straight full-bleed crop with no padding (the original
    # behavior, and default). Clamped to the source width.
    balanced = cfg.reframe_style == "balanced"
    active_w = min(w, int(crop_w * 1.35)) // 2 * 2 if balanced else crop_w

    if len(kfs) >= 2:
        x_part = f"x='{_crop_x_mixed_expr(kfs, active_w, w)}'"
        log.info("speaker crop: %d keyframes (glide within shots, snap at cuts)",
                 len(kfs))
    else:
        cx = kfs[0][1] if kfs else 0.5
        if kfs:
            log.info("face-aware crop (static): center x = %.2f", cx)
        x = max(0, min(int(cx * w - active_w / 2), w - active_w))
        x_part = f"x={x}"

    if not balanced:
        return (f"crop=w={crop_w}:h={h}:{x_part}:y=0,"
                f"scale={cfg.out_width}:{cfg.out_height},setsar=1")

    # multi-chain graph: split into a blurred full-frame background and a
    # less-tightly-cropped foreground, then composite. Valid to splice in
    # here because the caller appends ",ass=...[v]" onto whatever this
    # returns -- overlay is the last (unlabeled) filterchain, so the comma
    # continuation and final [v] label land correctly either way.
    return (
        f"split=2[bg0][fg0];"
        f"[bg0]scale={cfg.out_width}:{cfg.out_height}:force_original_aspect_ratio=increase,"
        f"crop={cfg.out_width}:{cfg.out_height},gblur=sigma=20[bg];"
        f"[fg0]crop=w={active_w}:h={h}:{x_part}:y=0,scale={cfg.out_width}:-2,setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )


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
