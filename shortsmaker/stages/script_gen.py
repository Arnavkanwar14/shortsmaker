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
    "no emoji, no hashtags, no markdown."
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
        "Write the narration: a hook in the first sentence that names what's "
        "concretely at stake, then react to and comment on what's happening. "
        f"Write {int(max_words * 0.75)}-{max_words} words (no fewer -- the "
        f"voiceover must fill most of the clip at about "
        f"{cfg.words_per_second} words/sec). Do NOT just repeat the "
        "dialogue -- add reaction and insight a viewer wouldn't think of. "
        "End with a line that makes the viewer want to comment or rewatch."
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

    duration = clip["end"] - clip["start"]
    max_words = int(duration * cfg.words_per_second)
    reply = llm.complete(cfg, _prompt(cfg, clip, duration, context or {}),
                         system=SYSTEM, max_tokens=max_words * 3)
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
