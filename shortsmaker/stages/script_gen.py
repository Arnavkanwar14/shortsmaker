"""STAGE 4 -- GENERATE COMMENTARY SCRIPT per selected window.

Uses the configured LLM (Ollama local by default, Groq free tier optional).
If no LLM is reachable, falls back to a template script built from the
clip's own transcript so the pipeline still completes.

Output per clip: clips/clip_NN/script.txt
"""
from __future__ import annotations

import logging
import re

from .. import llm
from ..config import Config

log = logging.getLogger("shortsmaker")

SYSTEM = ("You write short-form voiceover scripts for TikTok/Reels/Shorts. "
          "Output ONLY the words to be spoken -- no stage directions, no "
          "quotes, no emoji, no hashtags, no markdown.")


def _prompt(cfg: Config, clip_text: str, duration: float) -> str:
    max_words = int(duration * cfg.words_per_second)
    return (
        "You are writing a short-form voiceover script to accompany this clip.\n"
        f"Clip transcript (original dialogue):\n\"\"\"\n{clip_text}\n\"\"\"\n\n"
        "Write a punchy hook for the first 3 seconds, then narrate/comment on "
        "what's happening in an energetic, TikTok-style tone. Write "
        f"{int(max_words * 0.75)}-{max_words} words (no fewer -- the voiceover "
        f"must fill most of the {duration:.0f}-second clip at about "
        f"{cfg.words_per_second} words/sec). Do NOT just repeat the original "
        "dialogue -- add reaction and insight. End with a line that makes the "
        "viewer want to comment or rewatch."
    )


def _clean(text: str) -> str:
    text = re.sub(r"^(here('s| is).*?:|script:|voiceover:)\s*", "", text,
                  flags=re.I | re.S)
    text = re.sub(r"[*_#>\[\]`]", "", text)          # strip markdown remnants
    text = re.sub(r"\(.*?\)", "", text)              # stage directions
    return re.sub(r"\s+", " ", text).strip().strip('"')


def _fallback_script(clip_text: str, max_words: int) -> str:
    """No-LLM template: hook + condensed original content + CTA."""
    sentences = re.split(r"(?<=[.!?])\s+", clip_text)
    body_budget = max(max_words - 16, 10)
    body, used = [], 0
    for s in sentences:
        w = len(s.split())
        if used + w > body_budget:
            break
        body.append(s)
        used += w
    return ("Wait for this -- you won't believe what happens here. "
            + " ".join(body)
            + " Watch it again and tell me you caught that the first time.")


def run(cfg: Config, clip: dict, clip_dir) -> str:
    out = clip_dir / "script.txt"
    if out.exists() and not cfg.force:
        log.info("script: %s exists, skipping", out)
        return out.read_text(encoding="utf-8")

    duration = clip["end"] - clip["start"]
    max_words = int(duration * cfg.words_per_second)
    reply = llm.complete(cfg, _prompt(cfg, clip["text"], duration), system=SYSTEM,
                         max_tokens=max_words * 3)
    script = _clean(reply) if reply else ""

    if not script:
        log.info("no LLM available -- using template fallback script")
        script = _fallback_script(clip["text"], max_words)

    words = script.split()
    if len(words) > int(max_words * 1.15):          # hard-trim overruns
        script = " ".join(words[:max_words])
        if not re.search(r"[.!?]$", script):
            script += "."
        log.info("script trimmed to %d words for %ds budget", max_words, int(duration))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    log.info("script (%d words): %s...", len(script.split()), script[:70])
    return script
