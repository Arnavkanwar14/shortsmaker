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

    # ---- LLM (script + optional highlight ranking) ----
    llm_provider: str = "auto"      # auto | ollama | groq | gemini | none
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    groq_model: str = "llama-3.3-70b-versatile"   # free tier, needs GROQ_API_KEY
    gemini_model: str = "gemini-2.0-flash"        # free tier, needs GEMINI_API_KEY
    words_per_second: float = 2.5   # script length budget

    # ---- TTS ----
    tts_engine: str = "edge"        # edge | piper
    voice: str = "en-US-GuyNeural"  # edge-tts voice
    piper_model: str = ""           # path to a .onnx piper voice, if using piper

    # ---- assemble ----
    style: str = "kinetic"          # kinetic | plain | none (captions style)
    out_width: int = 1080
    out_height: int = 1920
    bg_audio_volume: float = 0.18   # duck original audio to ~18%
    face_crop: bool = True          # mediapipe face-aware crop, else center

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


def groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
