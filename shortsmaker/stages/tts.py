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


# --------------------------------------------------------------- kokoro
_KOKORO = {}


def _kokoro_pipeline(voice: str):
    """Cache one KPipeline per language (model load is the slow part)."""
    lang = voice[0] if voice[:1] in ("a", "b") else "a"   # af_/am_=US, bf_/bm_=UK
    if lang not in _KOKORO:
        from kokoro import KPipeline
        log.info("loading Kokoro-82M (%s english) -- one-time model download "
                 "on first use ...", "american" if lang == "a" else "british")
        _KOKORO[lang] = KPipeline(lang_code=lang)
    return _KOKORO[lang]


def synth_kokoro(cfg: Config, text: str, out: Path, speed: float = 1.0) -> list[dict]:
    """Local Kokoro-82M: the best-rated open TTS, small enough for CPU.
    Word timestamps come from Kokoro's own token alignment; whisper
    forced-alignment is the fallback."""
    import numpy as np
    import soundfile as sf

    pipe = _kokoro_pipeline(cfg.kokoro_voice)
    wav = out.with_suffix(".wav")
    parts, words = [], []
    offset = 0.0
    for result in pipe(text, voice=cfg.kokoro_voice, speed=speed):
        audio = result.audio
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        for t in (getattr(result, "tokens", None) or []):
            ts, te = getattr(t, "start_ts", None), getattr(t, "end_ts", None)
            text = t.text.strip()
            if ts is None or not text:
                continue
            if not any(ch.isalnum() for ch in text):
                # punctuation tokens get their own timestamps -- glue them
                # to the previous word instead of becoming caption "words"
                if words:
                    words[-1]["text"] += text
                continue
            words.append({"start": round(offset + ts, 3),
                          "end": round(offset + (te if te is not None else ts + 0.2), 3),
                          "text": text})
        parts.append(audio)
        offset += len(audio) / 24000
    if not parts:
        raise RuntimeError("kokoro produced no audio")
    sf.write(str(wav), np.concatenate(parts), 24000)
    run_ffmpeg(["-i", str(wav), "-b:a", "192k", str(out)])
    if not words:
        words = _align_with_whisper(cfg, wav)
    wav.unlink(missing_ok=True)
    return words


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

    # Cap how much we'll ever speed up the voice to fit -- a fast, rushed
    # voiceover is worse than one that overruns the clip by a beat (the
    # background bed is padded via apad, so a slight overrun just plays a
    # touch of trailing audio rather than cutting mid-word). 1.10 is
    # barely perceptible; the old 1.25 cap was a real, audible speed-up
    # and the actual cause of voiceovers sounding rushed.
    MAX_VO_SPEEDUP = 1.10

    def _edge_with_fit() -> list[dict]:
        w = synth_edge(cfg, script, audio)
        vo_len = media_duration(audio)
        if vo_len > clip_len - 0.3:
            speedup = min(vo_len / max(clip_len - 0.5, 1), MAX_VO_SPEEDUP)
            rate = f"+{int((speedup - 1) * 100)}%"
            log.info("VO %.1fs > clip %.1fs -- retrying at rate %s", vo_len, clip_len, rate)
            w = synth_edge(cfg, script, audio, rate=rate)
        return w

    if cfg.tts_engine == "piper":
        words = synth_piper(cfg, script, audio)
    elif cfg.tts_engine == "kokoro":
        try:
            words = synth_kokoro(cfg, script, audio)
            vo_len = media_duration(audio)
            if vo_len > clip_len - 0.3:
                speed = round(min(vo_len / max(clip_len - 0.5, 1), MAX_VO_SPEEDUP), 2)
                log.info("VO %.1fs > clip %.1fs -- retrying at speed %.2fx",
                         vo_len, clip_len, speed)
                words = synth_kokoro(cfg, script, audio, speed=speed)
        except Exception as e:
            log.warning("kokoro TTS failed (%s); falling back to edge-tts",
                        str(e)[:150])
            words = _edge_with_fit()
    else:
        words = _edge_with_fit()

    if not words:
        raise RuntimeError("TTS produced no word timestamps")
    write_json(words_file, words)
    log.info("voiceover: %.1fs audio, %d words (%s)",
             media_duration(audio), len(words), cfg.tts_engine)
    return audio, words
