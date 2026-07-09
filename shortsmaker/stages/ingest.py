"""STAGE 1 -- INGEST: accept a file or URL, produce normalized.mp4."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

from ..config import Config
from ..util import ffmpeg_location_for_ytdlp, ffprobe_video, run_ffmpeg, write_json

log = logging.getLogger("shortsmaker")

# Same pattern as highlights_signature/render_signature: normalized.mp4 was
# cached purely by "does the file exist", so changing max_height or
# target_fps on a URL you'd already run silently kept the old normalization.
INGEST_SETTINGS_KEYS = ["max_height", "target_fps"]


def ingest_signature(cfg: Config) -> str:
    d = {k: getattr(cfg, k) for k in INGEST_SETTINGS_KEYS}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")


def download(url: str, dest_dir: Path, max_height: int = 1080) -> Path:
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_dir / "source.%(ext)s")
    # cap at max_height: we normalize down to it anyway, so 4K sources just
    # waste bandwidth and make the re-encode dramatically slower
    fmt = (f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
           f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best")
    opts = {
        "format": fmt,
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
        info = ydl.extract_info(url, download=True)
    write_json(dest_dir / "meta.json", {
        "title": info.get("title", ""),
        "uploader": info.get("uploader", ""),
        "description": (info.get("description") or "")[:500],
    })
    files = sorted(dest_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp reported success but no file was written")
    return files[0]


def run(cfg: Config) -> Path:
    """Returns path to normalized.mp4 in the run dir."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized = run_dir / "normalized.mp4"
    sig = ingest_signature(cfg)
    sig_file = run_dir / "ingest_sig.txt"
    stale = not sig_file.is_file() or sig_file.read_text(encoding="utf-8").strip() != sig
    if normalized.exists() and not cfg.force and not stale:
        log.info("ingest: %s exists, skipping (use --force to redo)", normalized.name)
        return normalized
    if stale and normalized.exists() and not cfg.force:
        log.info("ingest: max_height/target_fps changed -- re-normalizing")

    if is_url(cfg.input):
        source = download(cfg.input, run_dir, cfg.max_height)
    else:
        src = Path(cfg.input)
        if not src.exists():
            raise FileNotFoundError(f"input file not found: {src}")
        source = run_dir / f"source{src.suffix.lower()}"
        if not source.exists():
            shutil.copy2(src, source)
        write_json(run_dir / "meta.json",
                   {"title": src.stem.replace("_", " ").replace("-", " "),
                    "uploader": "", "description": ""})

    info = ffprobe_video(source)
    log.info("source: %dx%d %s, %.1fs, %.1f fps", info["width"], info["height"],
             info.get("codec", "?"), info["duration"], info["fps"])

    # already-compliant sources (the common case now that downloads are
    # capped at 1080p h264) just get remuxed -- seconds instead of minutes
    compliant = (info.get("codec") == "h264"
                 and info["height"] <= cfg.max_height
                 and info["fps"] <= 61
                 and source.suffix.lower() == ".mp4")
    if compliant:
        log.info("source already compliant -- remuxing without re-encode")
        run_ffmpeg(["-i", str(source), "-c", "copy",
                    "-movflags", "+faststart", str(normalized)])
        sig_file.write_text(sig, encoding="utf-8")
        return normalized

    vf = []
    if info["height"] > cfg.max_height:
        vf.append(f"scale=-2:{cfg.max_height}")
    args = ["-i", str(source), "-r", str(cfg.target_fps)]
    if vf:
        args += ["-vf", ",".join(vf)]
    # veryfast: this is an intermediate file -- the final clip encode sets
    # the delivered quality, so spending encode time here buys nothing
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
             str(normalized)]
    log.info("normalizing to %d fps mp4 ...", cfg.target_fps)
    run_ffmpeg(args)
    sig_file.write_text(sig, encoding="utf-8")
    return normalized
