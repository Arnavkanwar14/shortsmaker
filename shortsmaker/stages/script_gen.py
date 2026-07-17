"""STAGE 4 -- GENERATE COMMENTARY SCRIPT per selected window.

Uses the configured LLM (Ollama local by default, Groq free tier optional).
The prompt carries the video's title and the dialogue just before/after the
clip so the narration reacts to what's actually happening instead of
guessing from an isolated snippet. If no LLM is reachable, falls back to a
template script built from the clip's own transcript.

Output per clip: clips/clip_NN/script.txt
"""
from __future__ import annotations

import logging
import re

from .. import llm
from ..config import Config

log = logging.getLogger("shortsmaker")

SYSTEM = (
    "You write voiceover scripts for viral TikTok/Reels/Shorts clips. "
    "Sound like a real person hyping a moment to a friend: conversational, "
    "contractions, short punchy sentences, present tense, specific to what's "
    "actually happening in THIS clip. Reference concrete details (names, "
    "numbers, objects, what someone says or does) -- never generic filler "
    "like 'you won't believe this' or 'wait for it'. "
    "Output ONLY the words to be spoken -- no stage directions, no quotes, "
    "no emoji, no hashtags, no markdown. "
    "ALWAYS write the script in English, even if the source dialogue you are "
    "given is in a different language -- translate/react in English, never "
    "mirror the source language. "
    "Never narrate ahead of the footage: only react to dialogue that has "
    "already happened at that point in the script, in the same order it "
    "occurs -- describing a later moment early is the #1 thing that makes "
    "a voiceover feel out of sync with the video."
)


def _prompt(cfg: Config, clip: dict, duration: float, context: dict) -> str:
    max_words = int(duration * cfg.words_per_second)
    parts = []
    if context.get("title"):
        parts.append(f"The video is titled: \"{context['title']}\"")
        if context.get("uploader"):
            parts[-1] += f" (channel: {context['uploader']})"
    if context.get("before_text"):
        parts.append(f"Dialogue just BEFORE this clip (for context only):\n"
                     f"{context['before_text']}")
    parts.append(f"The clip's own dialogue:\n\"\"\"\n{clip['text'] or '(no speech -- action footage)'}\n\"\"\"")
    if context.get("after_text"):
        parts.append(f"Dialogue just AFTER this clip (for context only):\n"
                     f"{context['after_text']}")
    ctx_block = "\n\n".join(parts)
    return (
        f"You are writing the voiceover for a {duration:.0f}-second vertical "
        f"short cut from a longer video.\n\n{ctx_block}\n\n"
        f"HARD LENGTH LIMIT: {int(max_words * 0.75)}-{max_words} words, no "
        f"more (about {cfg.words_per_second} words/sec for a "
        f"{duration:.0f}-second clip). This is a strict budget you must plan "
        "against as you write -- if the dialogue has many beats, compress or "
        "skip the less important/repetitive early ones (e.g. summarize a "
        "grind or setup in one line) so you have enough words left to reach "
        "and clearly describe the clip's ENDING. The final beat/payoff/twist "
        "in the dialogue above is the most important thing to land -- a "
        "script that runs out of words before describing it has failed, "
        "even if every earlier detail was covered.\n\n"
        "Write the narration: a hook in the first sentence that grabs "
        "attention using ONLY what has already happened in the dialogue so "
        "far -- never reference, describe, or foreshadow something that "
        "occurs LATER in the dialogue above (no spoiling ahead of the "
        "footage). Then react to and comment on events in the SAME order "
        "they occur in the dialogue, ending on a reaction to the clip's "
        "final moment. Do NOT just repeat the dialogue -- add reaction and "
        "insight a viewer wouldn't think of. End with a short line reacting "
        "to that final moment that makes the viewer want to comment or "
        "rewatch -- do not add anything after it. "
        "Write the script in ENGLISH regardless of what language the clip's "
        "own dialogue above is in."
    )


def _clean(text: str) -> str:
    text = re.sub(r"^(here('s| is).*?:|script:|voiceover:)\s*", "", text,
                  flags=re.I | re.S)
    text = re.sub(r"[*_#>\[\]`]", "", text)          # strip markdown remnants
    text = re.sub(r"\(.*?\)", "", text)              # stage directions
    return re.sub(r"\s+", " ", text).strip().strip('"')


def _fallback_script(clip_text: str, max_words: int) -> str:
    """No-LLM template: condensed original content with light framing."""
    sentences = re.split(r"(?<=[.!?])\s+", clip_text)
    body_budget = max(max_words - 12, 10)
    body, used = [], 0
    for s in sentences:
        w = len(s.split())
        if used + w > body_budget:
            break
        body.append(s)
        used += w
    return ("Okay, watch this closely. " + " ".join(body)
            + " Tell me you caught that the first time.")


def clip_context(transcript: dict, clip: dict, meta: dict,
                 context_words: int = 40) -> dict:
    """Video title + the dialogue surrounding the clip window."""
    before, after = [], []
    for seg in transcript.get("segments", []):
        if seg["end"] <= clip["start"]:
            before.append(seg["text"])
        elif seg["start"] >= clip["end"]:
            after.append(seg["text"])
    return {
        "title": meta.get("title", ""),
        "uploader": meta.get("uploader", ""),
        "before_text": " ".join(" ".join(before).split()[-context_words:]),
        "after_text": " ".join(" ".join(after).split()[:context_words]),
    }


def run(cfg: Config, clip: dict, clip_dir, context: dict | None = None) -> str:
    out = clip_dir / "script.txt"
    if out.exists() and not cfg.force:
        log.info("script: %s exists, skipping", out)
        return out.read_text(encoding="utf-8")

    # budget against the post-cut length when dead air was trimmed
    duration = clip.get("edited_duration") or (clip["end"] - clip["start"])
    max_words = int(duration * cfg.words_per_second)
    reply = llm.complete(cfg, _prompt(cfg, clip, duration, context or {}),
                         system=SYSTEM, max_tokens=max_words * 3)
    script = _clean(reply) if reply else ""

    if not script:
        log.info("no LLM available -- using template fallback script")
        script = _fallback_script(clip["text"], max_words)

    # tts.py's own fit-check absorbs a modest overrun with a barely-audible
    # speed-up (capped at 1.10x) plus a harmless padded tail, so this is a
    # safety net for runaway replies only -- not the primary length control
    # (that's the prompt's own word budget). Loosened from 1.15x: at the
    # tighter threshold this trim frequently sliced off the clip's climax,
    # which the model -- writing in chronological order -- always puts last.
    words = script.split()
    overrun_limit = int(max_words * 1.6)
    if len(words) > overrun_limit:
        # trim at the last full sentence at/before the limit, not mid-sentence
        truncated = " ".join(words[:overrun_limit])
        sentences = re.findall(r".*?[.!?](?:\s|$)", truncated, flags=re.S)
        script = "".join(sentences).strip() if sentences else truncated + "."
        log.info("script trimmed to ~%d words (overran %d-word cap for %ds budget)",
                 len(script.split()), overrun_limit, int(duration))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    log.info("script (%d words): %s...", len(script.split()), script[:70])
    return script
