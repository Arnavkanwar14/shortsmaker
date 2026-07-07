"""STAGE 2 -- TRANSCRIBE: faster-whisper, word-level timestamps.

Output: transcript.json
  {"language": ..., "segments": [{start, end, text, words: [{start, end, text}]}]}
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from ..config import Config
from ..util import extract_wav, read_json, write_json

log = logging.getLogger("shortsmaker")


def _register_cuda_dlls() -> None:
    """pip-installed nvidia wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12)
    put their DLLs under site-packages/nvidia/*/bin. Windows won't find
    them unless the dirs are on the DLL search path before ctranslate2
    loads -- this is what makes GPU whisper work without a full CUDA
    Toolkit install."""
    import os
    import site
    if os.name != "nt":
        return
    for sp in site.getsitepackages():
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for bin_dir in nv.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            # ctranslate2 loads cublas/cudnn with a plain LoadLibrary call,
            # which searches PATH -- add_dll_directory alone isn't seen
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def run(cfg: Config, video: Path) -> dict:
    out = cfg.run_dir / "transcript.json"
    if out.exists() and not cfg.force:
        log.info("transcribe: transcript.json exists, skipping")
        return read_json(out)

    _register_cuda_dlls()
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
        # batched inference is markedly faster on both CPU and GPU with
        # identical output quality; fall back if this faster-whisper
        # version doesn't support it (or rejects an argument)
        try:
            from faster_whisper import BatchedInferencePipeline
            segs, info = BatchedInferencePipeline(model).transcribe(
                str(wav), word_timestamps=True, batch_size=8)
            return list(segs), info
        except Exception as e:
            log.info("batched inference unavailable (%s); using sequential",
                     str(e)[:120])
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

    segments = _sentence_segments(segments)
    transcript = {"language": info.language, "segments": segments}
    write_json(out, transcript)
    log.info("transcribed %d segments (language=%s)", len(segments), info.language)
    return transcript


def _sentence_segments(segments: list[dict], max_words: int = 40) -> list[dict]:
    """Rebuild segments at sentence granularity from word timestamps.
    Batched whisper returns coarse ~30s chunks, which would blunt the
    highlight windowing; sentence-level segments keep it sharp regardless
    of which inference path produced them."""
    words = [w for s in segments for w in s.get("words", [])]
    if not words:
        return segments
    out, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?…]$", w["text"]) or len(cur) >= max_words:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                        "text": " ".join(x["text"] for x in cur), "words": cur})
            cur = []
    if cur:
        out.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                    "text": " ".join(x["text"] for x in cur), "words": cur})
    return out
