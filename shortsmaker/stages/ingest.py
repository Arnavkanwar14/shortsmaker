"""STAGE 1 -- INGEST: accept a file or URL, produce normalized.mp4."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

from ..config import Config
from ..util import ffmpeg_location_for_ytdlp, ffprobe_video, run_ffmpeg

log = logging.getLogger("shortsmaker")


def is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")


def download(url: str, dest_dir: Path) -> Path:
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_dir / "source.%(ext)s")
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_tmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    ffloc = ffmpeg_location_for_ytdlp()
    if ffloc:
        opts["ffmpeg_location"] = ffloc
    log.info("downloading %s ...", url)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = sorted(dest_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp reported success but no file was written")
    return files[0]


def run(cfg: Config) -> Path:
    """Returns path to normalized.mp4 in the run dir."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized = run_dir / "normalized.mp4"
    if normalized.exists() and not cfg.force:
        log.info("ingest: %s exists, skipping (use --force to redo)", normalized.name)
        return normalized

    if is_url(cfg.input):
        source = download(cfg.input, run_dir)
    else:
        src = Path(cfg.input)
        if not src.exists():
            raise FileNotFoundError(f"input file not found: {src}")
        source = run_dir / f"source{src.suffix.lower()}"
        if not source.exists():
            shutil.copy2(src, source)

    info = ffprobe_video(source)
    log.info("source: %dx%d, %.1fs, %.1f fps", info["width"], info["height"],
             info["duration"], info["fps"])

    vf = []
    if info["height"] > cfg.max_height:
        vf.append(f"scale=-2:{cfg.max_height}")
    args = ["-i", str(source), "-r", str(cfg.target_fps)]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
             str(normalized)]
    log.info("normalizing to %d fps mp4 ...", cfg.target_fps)
    run_ffmpeg(args)
    return normalized
