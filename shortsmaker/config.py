"""Central configuration for the shortsmaker pipeline.

Every model/engine choice lives here so a paid option can be swapped in
later by editing a config file or passing CLI flags -- no code changes.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- run ----
    input: str = ""                 # local path or URL
    workdir: str = "runs"           # parent folder for per-run working dirs
    run_id: str = ""                # derived from input if empty
    num_clips: int = 5
    duration: int = 45              # target clip length, seconds
    min_duration: int = 30
    max_duration: int = 60

    # ---- ingest ----
    target_fps: int = 30
    max_height: int = 1080          # normalize tall sources down to this

    # ---- transcribe ----
    whisper_model: str = "small"    # tiny/base/small/medium/large-v3
    whisper_device: str = "auto"    # auto/cpu/cuda
    whisper_compute: str = "auto"   # auto/int8/float16

    # ---- highlights ----
    scene_threshold: float = 27.0   # PySceneDetect ContentDetector threshold
    use_llm_highlights: bool = True # only if an LLM provider is reachable
    content_type: str = "auto"      # auto | talk | action | funny
                                    # action: gaming/sports -- scores cut
                                    # density + loudness bursts over dialogue
    focus: str = ""                 # optional: keywords or a description of
                                    # the moments wanted ("boss fight",
                                    # "funny fails", "when they talk pricing")
    manual_clips: str = ""          # optional: explicit spans, skips auto
                                    # detection: "12:30-13:10, 745-790"

    # ---- LLM (script + optional highlight ranking) ----
    llm_provider: str = "auto"      # auto | ollama | groq | gemini | none
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    groq_model: str = "llama-3.3-70b-versatile"   # free tier, needs GROQ_API_KEY
    gemini_model: str = "gemini-2.0-flash"        # free tier, needs GEMINI_API_KEY
    words_per_second: float = 2.5   # script length budget

    # ---- TTS / voiceover ----
    voiceover: bool = True          # off: keep original audio, captions come
                                    # from the source speech instead of TTS
    vo_volume: float = 1.0          # voiceover loudness (0..1+)
    tts_engine: str = "edge"        # edge | kokoro | piper
    kokoro_voice: str = "af_heart"  # local Kokoro-82M voice (best open TTS;
                                    # runs on CPU, falls back to edge-tts)
    voice: str = "en-US-AndrewMultilingualNeural"  # edge-tts voice; the
                                    # *MultilingualNeural voices are the newest
                                    # generation and sound far less robotic
    piper_model: str = ""           # path to a .onnx piper voice, if using piper

    # ---- editing ----
    trim_silence: str = "auto"      # auto | on | off -- cut dead air and
                                    # filler words (um/uh) for snappy pacing.
                                    # auto = on for speech-driven clips.
    silence_gap: float = 0.4        # silences longer than this get cut (s)

    # ---- assemble ----
    style: str = "kinetic"          # kinetic | plain | none (captions style)
    caption_preset: str = "bold"    # bold | beast | minimal | karaoke
    caption_position: str = "lower" # lower | center
    out_width: int = 1080
    out_height: int = 1920
    bg_audio_volume: float = -1.0   # original-audio volume; -1 = auto
                                    # (ducked to 0.18 under a voiceover,
                                    #  full 1.0 when voiceover is off)
    face_crop: bool = True          # mediapipe face-aware crop, else center
    reframe_style: str = "tight"    # tight | balanced (less zoom, blurred
                                    # top/bottom fill -- see assemble.crop_filter)

    # ---- cleanup (stage 7) ----
    clean: bool = False             # inpaint burned-in captions/watermarks

    # ---- misc ----
    force: bool = False             # re-run stages even if outputs exist
    stages: list[str] = field(default_factory=lambda: [
        "ingest", "transcribe", "highlights", "script", "tts", "assemble",
    ])

    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.workdir) / self.run_id

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def apply_overrides(self, overrides: dict) -> None:
        for k, v in overrides.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)


def _load_dotenv() -> None:
    """Load KEY=value lines from a .env file (project root, then cwd) into
    os.environ, without overriding variables that are already set."""
    for candidate in (Path(__file__).resolve().parent.parent / ".env",
                      Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()


def groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


def hf_token() -> str:
    return os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")


def load_channel() -> dict:
    """Optional per-channel metadata constants from channel.json in the
    project root. Generalizes the branded/fixed parts of upload metadata
    (subscribe CTA, always-include tags, fixed hashtags) so the SEO rules
    aren't hardcoded to any one channel. Empty dict = pure generic SEO.
    Recognized keys: name, cta, tags (list), hashtags (list)."""
    path = Path(__file__).resolve().parent.parent / "channel.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
