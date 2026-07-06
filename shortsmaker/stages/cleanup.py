"""STAGE 7 -- CLEANUP (optional, --clean): remove burned-in captions or
watermarks from the source before assembly, using a locally-run
open-source inpainting model (LaMa via the `iopaint` package).

This is by far the most compute-heavy stage, so it is skipped by default.
It requires: pip install iopaint   (and ideally a GPU).

Approach: export frames -> iopaint batch inpainting with a static mask
covering the region given by --clean-box -> re-encode with original audio.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ..config import Config
from ..util import ffprobe_video, run_ffmpeg

log = logging.getLogger("shortsmaker")


def run(cfg: Config, video: Path, box: tuple[int, int, int, int] | None = None) -> Path:
    """box = (x, y, w, h) region to remove; defaults to the bottom 18%
    of the frame (typical burned-caption band)."""
    out = cfg.run_dir / "cleaned.mp4"
    if out.exists() and not cfg.force:
        log.info("cleanup: cleaned.mp4 exists, skipping")
        return out

    if shutil.which("iopaint") is None:
        log.warning("cleanup requested but `iopaint` is not installed "
                    "(pip install iopaint). Skipping stage 7.")
        return video

    info = ffprobe_video(video)
    w, h = info["width"], info["height"]
    if box is None:
        box = (0, int(h * 0.82), w, h - int(h * 0.82))

    work = cfg.run_dir / "clean_work"
    frames, fixed = work / "frames", work / "fixed"
    frames.mkdir(parents=True, exist_ok=True)
    fixed.mkdir(parents=True, exist_ok=True)

    log.info("cleanup: extracting frames (this is slow and disk-hungry) ...")
    run_ffmpeg(["-i", str(video), str(frames / "f%06d.png")])

    # static mask: white = region to inpaint
    mask = work / "mask.png"
    _write_mask(mask, w, h, box)

    log.info("cleanup: running LaMa inpainting via iopaint on %d frames ...",
             len(list(frames.glob("*.png"))))
    subprocess.run(
        ["iopaint", "run", "--model=lama", "--device=cpu",
         f"--image={frames}", f"--mask={mask}", f"--output={fixed}"],
        check=True,
    )

    log.info("cleanup: re-encoding with original audio ...")
    run_ffmpeg(["-framerate", str(cfg.target_fps), "-i", str(fixed / "f%06d.png"),
                "-i", str(video), "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "copy", str(out)])
    shutil.rmtree(work, ignore_errors=True)
    return out


def _write_mask(path: Path, w: int, h: int, box: tuple[int, int, int, int]) -> None:
    import numpy as np
    try:
        import cv2
        m = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = box
        m[y:y + bh, x:x + bw] = 255
        cv2.imwrite(str(path), m)
    except ImportError:
        from PIL import Image
        img = Image.new("L", (w, h), 0)
        x, y, bw, bh = box
        img.paste(255, (x, y, x + bw, y + bh))
        img.save(path)
