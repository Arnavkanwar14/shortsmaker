"""STAGE 3 -- FIND THE BEST MOMENTS.

Combines several free signals (no single one is trusted alone):
  - PySceneDetect shot boundaries (windows that start on a cut feel cleaner)
  - audio RMS energy percentiles via librosa (loud/energetic passages)
  - speech-rate deviation (fast excited talking or dramatic slow-downs)
  - keyword/emotion density (questions, superlatives, numbers, laughter)
  - sentence completeness at the window edges
Optionally re-ranked by a free LLM pass over the timestamped transcript.

Output: highlights.json -- ranked [{start, end, score, reason, signals}]
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .. import llm
from ..config import Config
from ..util import read_json, write_json

log = logging.getLogger("shortsmaker")

SUPERLATIVES = re.compile(
    r"\b(best|worst|most|least|biggest|smallest|craziest|insane|amazing|"
    r"incredible|unbelievable|never|always|literally|actually|secret|"
    r"nobody|everyone|huge|massive|shocking|wild|epic|perfect|terrible|"
    r"free|hack|trick|mistake|wrong|right|truth|real|fake)\b", re.I)
LAUGHTER = re.compile(r"\b(ha(ha)+|lol|lmao)\b|\[laugh", re.I)
NUMBERS = re.compile(r"\b\d[\d,.]*%?\b|\b(million|billion|thousand|hundred)\b", re.I)
SENTENCE_END = re.compile(r"[.!?…]\s*$")


# ------------------------------------------------------------- signals
def detect_scenes(video: Path, threshold: float) -> list[float]:
    """Shot-change timestamps in seconds."""
    try:
        from scenedetect import ContentDetector, detect
        scene_list = detect(str(video), ContentDetector(threshold=threshold))
        cuts = [s[0].get_seconds() for s in scene_list]
        log.info("scene detection: %d shots", len(cuts))
        return cuts
    except Exception as e:
        log.warning("scene detection failed (%s); continuing without cuts", e)
        return []


def audio_energy(wav: Path):
    """Return (times, rms) arrays for the whole audio track."""
    import librosa
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return times, rms


def window_energy_pct(times, rms, start: float, end: float) -> float:
    """Mean RMS of the window expressed as a percentile of the full track."""
    mask = (times >= start) & (times <= end)
    if not mask.any() or rms.max() <= 0:
        return 0.5
    mean = float(rms[mask].mean())
    return float((rms < mean).mean())   # fraction of track quieter than this window


def keyword_score(text: str) -> float:
    words = max(len(text.split()), 1)
    hits = (len(SUPERLATIVES.findall(text)) + len(NUMBERS.findall(text))
            + 2 * len(LAUGHTER.findall(text)) + text.count("?") + text.count("!"))
    return min(hits / (words / 25.0 + 1e-9) / 5.0, 1.0)   # ~hits per 25 words, capped


def completeness_score(first_text: str, last_text: str) -> float:
    score = 0.0
    if first_text[:1].isupper() or first_text[:1] in "\"'":
        score += 0.5
    if SENTENCE_END.search(last_text):
        score += 0.5
    return score


# ----------------------------------------------------------- windowing
def build_windows(segments: list[dict], cfg: Config) -> list[dict]:
    """Grow candidate windows from each segment start until min..max duration."""
    windows = []
    n = len(segments)
    for i in range(n):
        start = segments[i]["start"]
        j = i
        while j < n and segments[j]["end"] - start < cfg.min_duration:
            j += 1
        while j < n and segments[j]["end"] - start <= cfg.max_duration:
            windows.append({"start": start, "end": segments[j]["end"],
                            "seg_range": (i, j)})
            # prefer stopping at sentence ends: keep only sentence-final
            # extensions after the first valid one
            if SENTENCE_END.search(segments[j]["text"]):
                break
            j += 1
    return windows


def snap_to_scene(t: float, cuts: list[float], tol: float = 1.5) -> tuple[float, bool]:
    for c in cuts:
        if abs(c - t) <= tol:
            return c, True
    return t, False


def score_windows(windows, segments, cuts, times, rms, cfg: Config) -> list[dict]:
    speech_rates = []
    for w in windows:
        i, j = w["seg_range"]
        text = " ".join(s["text"] for s in segments[i:j + 1])
        dur = w["end"] - w["start"]
        speech_rates.append(len(text.split()) / max(dur, 1))
    mean_rate = float(np.mean(speech_rates)) if speech_rates else 1.0
    std_rate = float(np.std(speech_rates)) or 1.0

    scored = []
    for w, rate in zip(windows, speech_rates):
        i, j = w["seg_range"]
        text = " ".join(s["text"] for s in segments[i:j + 1])
        start, on_cut = snap_to_scene(w["start"], cuts)
        signals = {
            "energy": window_energy_pct(times, rms, w["start"], w["end"]),
            "keywords": keyword_score(text),
            "speech_rate_dev": min(abs(rate - mean_rate) / std_rate / 2.0, 1.0),
            "completeness": completeness_score(segments[i]["text"], segments[j]["text"]),
            "scene_aligned": 1.0 if on_cut else 0.0,
        }
        score = (0.30 * signals["energy"] + 0.30 * signals["keywords"]
                 + 0.10 * signals["speech_rate_dev"]
                 + 0.20 * signals["completeness"] + 0.10 * signals["scene_aligned"])
        scored.append({
            "start": round(start, 2), "end": round(w["end"], 2),
            "score": round(score, 4), "signals": {k: round(v, 3) for k, v in signals.items()},
            "text": text,
            "reason": "heuristic: " + (", ".join(
                k for k, v in signals.items() if v >= 0.6) or "mixed signals"),
        })
    return scored


def pick_non_overlapping(scored: list[dict], k: int) -> list[dict]:
    scored = sorted(scored, key=lambda w: w["score"], reverse=True)
    picked: list[dict] = []
    for w in scored:
        if len(picked) >= k:
            break
        if all(min(w["end"], p["end"]) - max(w["start"], p["start"])
               < 0.2 * (w["end"] - w["start"]) for p in picked):
            picked.append(w)
    return sorted(picked, key=lambda w: w["start"])


# ------------------------------------------------------------ LLM pass
def llm_rerank(cfg: Config, segments: list[dict], candidates: list[dict]) -> list[dict]:
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    transcript_block = "\n".join(lines)
    if len(transcript_block) > 24000:            # keep local models happy
        transcript_block = transcript_block[:24000]
    prompt = (
        "Here is a timestamped transcript of a video.\n\n"
        f"{transcript_block}\n\n"
        f"Identify the {cfg.num_clips} most self-contained, high-interest "
        f"{cfg.min_duration}-{cfg.max_duration} second segments that would work as "
        "standalone short clips (strong hook, complete thought, emotional or "
        "surprising content). Respond with ONLY a JSON array like: "
        '[{"start": 12.5, "end": 55.0, "reason": "one line"}]'
    )
    reply = llm.complete(cfg, prompt, system="You find viral moments in videos.")
    picks = llm.extract_json_array(reply)
    if not picks:
        log.info("LLM highlight pass unavailable/unparseable; using heuristics only")
        return candidates
    boosted = []
    for c in candidates:
        c = dict(c)
        for p in picks:
            try:
                ps, pe = float(p["start"]), float(p["end"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(c["end"], pe) - max(c["start"], ps)
            if overlap > 0.5 * (c["end"] - c["start"]):
                c["score"] = round(c["score"] + 0.5, 4)
                c["reason"] = f"LLM: {p.get('reason', 'selected by LLM')}"
                break
        boosted.append(c)
    log.info("LLM highlight pass boosted %d candidate windows",
             sum(1 for b in boosted if b["reason"].startswith("LLM")))
    return boosted


# ---------------------------------------------------------------- main
def run(cfg: Config, video: Path, transcript: dict) -> list[dict]:
    out = cfg.run_dir / "highlights.json"
    if out.exists() and not cfg.force:
        log.info("highlights: highlights.json exists, skipping")
        return read_json(out)

    segments = [s for s in transcript["segments"] if s["text"].strip()]
    if not segments:
        raise RuntimeError("transcript is empty -- nothing to clip")

    cuts = detect_scenes(video, cfg.scene_threshold)
    wav = cfg.run_dir / "audio_16k.wav"
    times, rms = audio_energy(wav)

    windows = build_windows(segments, cfg)
    log.info("built %d candidate windows", len(windows))
    if not windows:  # very short source: take everything
        windows = [{"start": segments[0]["start"], "end": segments[-1]["end"],
                    "seg_range": (0, len(segments) - 1)}]

    scored = score_windows(windows, segments, cuts, times, rms, cfg)
    if cfg.use_llm_highlights:
        scored = llm_rerank(cfg, segments, scored)

    picked = pick_non_overlapping(scored, cfg.num_clips)
    for w in picked:
        log.info("pick %.1f-%.1fs score=%.2f (%s)", w["start"], w["end"],
                 w["score"], w["reason"][:60])
    write_json(out, picked)
    return picked
