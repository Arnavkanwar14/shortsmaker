"""STAGE 2 -- TRANSCRIBE: faster-whisper, word-level timestamps.

Output: transcript.json
  {"language": ..., "segments": [{start, end, text, words: [{start, end, text}]}]}
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config
from ..util import extract_wav, read_json, write_json

log = logging.getLogger("shortsmaker")


def run(cfg: Config, video: Path) -> dict:
    out = cfg.run_dir / "transcript.json"
    if out.exists() and not cfg.force:
        log.info("transcribe: transcript.json exists, skipping")
        return read_json(out)

    from faster_whisper import WhisperModel

    wav = cfg.run_dir / "audio_16k.wav"
    if not wav.exists() or cfg.force:
        extract_wav(video, wav)

    device = cfg.whisper_device
    compute = cfg.whisper_compute
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    def _transcribe(device: str, compute: str):
        log.info("loading whisper '%s' on %s (%s) ...", cfg.whisper_model, device, compute)
        model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute)
        segs, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True)
        return list(segs), info

    try:
        segments_iter, info = _transcribe(device, compute)
    except RuntimeError as e:
        if device == "cuda":
            # GPU visible but CUDA runtime libs missing/broken -> CPU fallback
            log.warning("CUDA transcription failed (%s); falling back to CPU", e)
            segments_iter, info = _transcribe("cpu", "int8")
        else:
            raise

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": [
                {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word.strip()}
                for w in (seg.words or [])
            ],
        })
        log.debug("  [%.1f-%.1f] %s", seg.start, seg.end, seg.text.strip()[:80])

    transcript = {"language": info.language, "segments": segments}
    write_json(out, transcript)
    log.info("transcribed %d segments (language=%s)", len(segments), info.language)
    return transcript
