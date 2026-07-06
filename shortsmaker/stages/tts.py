"""STAGE 5 -- VOICEOVER.

Default: edge-tts (free Microsoft neural voices, no API key). Its stream
emits WordBoundary events, giving word timestamps for free.
Fallback: piper (fully offline); timestamps recovered by force-aligning
the generated audio with faster-whisper.

Output per clip: voiceover.mp3 + vo_words.json [{start, end, text}]
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from ..config import Config
from ..util import media_duration, read_json, run_ffmpeg, write_json

log = logging.getLogger("shortsmaker")


# ------------------------------------------------------------- edge-tts
async def _edge_synth(text: str, voice: str, rate: str, out: Path) -> list[dict]:
    import edge_tts
    words = []
    try:
        # edge-tts >= 7 defaults to SentenceBoundary; ask for word events
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    except TypeError:  # edge-tts 6.x: word boundaries are the only mode
        comm = edge_tts.Communicate(text, voice, rate=rate)
    with open(out, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7            # 100-ns ticks -> s
                words.append({"start": round(start, 3),
                              "end": round(start + chunk["duration"] / 1e7, 3),
                              "text": chunk["text"]})
    return words


def synth_edge(cfg: Config, text: str, out: Path, rate: str = "+0%") -> list[dict]:
    return asyncio.run(_edge_synth(text, cfg.voice, rate, out))


# ---------------------------------------------------------------- piper
def synth_piper(cfg: Config, text: str, out: Path) -> list[dict]:
    if not cfg.piper_model:
        raise RuntimeError("tts_engine=piper requires --piper-model <voice.onnx>")
    wav = out.with_suffix(".wav")
    subprocess.run(
        ["piper", "--model", cfg.piper_model, "--output_file", str(wav)],
        input=text.encode("utf-8"), check=True,
    )
    run_ffmpeg(["-i", str(wav), "-b:a", "192k", str(out)])
    return _align_with_whisper(cfg, wav)


def _align_with_whisper(cfg: Config, wav: Path) -> list[dict]:
    """Forced alignment: transcribe our own TTS audio for word timestamps."""
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(wav), word_timestamps=True)
    words = []
    for seg in segs:
        for w in seg.words or []:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                          "text": w.word.strip()})
    return words


# ----------------------------------------------------------------- main
def run(cfg: Config, script: str, clip: dict, clip_dir: Path) -> tuple[Path, list[dict]]:
    audio = clip_dir / "voiceover.mp3"
    words_file = clip_dir / "vo_words.json"
    if audio.exists() and words_file.exists() and not cfg.force:
        log.info("tts: voiceover exists, skipping")
        return audio, read_json(words_file)

    # compare against the post-cut length when dead air was trimmed
    clip_len = clip.get("edited_duration") or (clip["end"] - clip["start"])

    if cfg.tts_engine == "piper":
        words = synth_piper(cfg, script, audio)
    else:
        words = synth_edge(cfg, script, audio)
        # if the VO overruns the clip, re-synthesize a bit faster (max +25%)
        vo_len = media_duration(audio)
        if vo_len > clip_len - 0.3:
            speedup = min(vo_len / max(clip_len - 0.5, 1), 1.25)
            rate = f"+{int((speedup - 1) * 100)}%"
            log.info("VO %.1fs > clip %.1fs -- retrying at rate %s", vo_len, clip_len, rate)
            words = synth_edge(cfg, script, audio, rate=rate)

    if not words:
        raise RuntimeError("TTS produced no word timestamps")
    write_json(words_file, words)
    log.info("voiceover: %.1fs audio, %d words (%s)",
             media_duration(audio), len(words), cfg.tts_engine)
    return audio, words
