"""Shared helpers: logging, ffmpeg resolution, subprocess wrappers,
manifest writing, and the SaaS cost-equivalent ledger.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("shortsmaker")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("faster_whisper", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ----------------------------------------------------------------- ffmpeg
_FFMPEG: str | None = None


def ffmpeg_exe() -> str:
    """Resolve ffmpeg: env var > PATH > bundled imageio-ffmpeg binary."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    import os
    cand = os.environ.get("SHORTSMAKER_FFMPEG") or shutil.which("ffmpeg")
    if not cand:
        try:
            import imageio_ffmpeg
            cand = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg, or `pip install imageio-ffmpeg`, "
                "or set SHORTSMAKER_FFMPEG to the binary path."
            )
    _FFMPEG = cand
    return cand


def ffmpeg_location_for_ytdlp() -> str | None:
    """yt-dlp needs an executable literally named `ffmpeg` to merge
    video+audio streams. If ffmpeg is not on PATH, copy the bundled
    imageio-ffmpeg binary (which has a versioned name) into a cache dir
    under the canonical name and return that dir; None if PATH has it."""
    import os
    if shutil.which("ffmpeg"):
        return None
    exe = Path(ffmpeg_exe())
    if exe.stem.lower() == "ffmpeg":
        return str(exe.parent)
    cache = Path.home() / ".cache" / "shortsmaker" / "ffmpeg"
    target = cache / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists() or target.stat().st_size != exe.stat().st_size:
        cache.mkdir(parents=True, exist_ok=True)
        log.info("staging bundled ffmpeg for yt-dlp (one-time copy) ...")
        shutil.copy2(exe, target)
    return str(cache)


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{res.stderr[-2000:]}")


def ffprobe_video(path: Path) -> dict:
    """Width/height/duration/fps via ffmpeg (no separate ffprobe needed)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    info: dict = {"width": 0, "height": 0, "duration": 0.0, "fps": 30.0}
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
    if m:
        h, mnt, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        info["duration"] = h * 3600 + mnt * 60 + s
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", res.stderr)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+\.?\d*)\s*fps", res.stderr)
    if m:
        info["fps"] = float(m.group(1))
    return info


def media_duration(path: Path) -> float:
    return ffprobe_video(path)["duration"]


def extract_wav(video: Path, wav: Path, sr: int = 16000) -> Path:
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), str(wav)])
    return wav


# ------------------------------------------------------------- json io
def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------- SaaS cost-equivalent
class CostLedger:
    """Tracks what a credit-based SaaS tool would have charged per stage,
    so the savings of the free pipeline are visible in the logs/manifest.
    Rates are rough public-pricing equivalents, not exact quotes.
    """

    RATES = {
        "transcribe": ("per source minute", 0.5),   # credits/min
        "highlights": ("per source minute", 0.3),
        "script":     ("per clip", 2.0),
        "tts":        ("per clip", 3.0),
        "assemble":   ("per clip", 5.0),
        "cleanup":    ("per clip", 10.0),
    }

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, stage: str, quantity: float) -> None:
        unit, rate = self.RATES.get(stage, ("", 0.0))
        credits = round(rate * quantity, 1)
        self.entries.append(
            {"stage": stage, "unit": unit, "quantity": round(quantity, 2),
             "saas_credits_equivalent": credits}
        )
        log.info("[cost] %s: ~%s SaaS credits equivalent (%s x %.2f) -- $0.00 here",
                 stage, credits, unit, quantity)

    @property
    def total(self) -> float:
        return round(sum(e["saas_credits_equivalent"] for e in self.entries), 1)

    def as_dict(self) -> dict:
        return {"entries": self.entries, "total_credits_equivalent": self.total}
