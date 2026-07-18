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
from ..util import write_json

log = logging.getLogger("shortsmaker")

# Clips at/above this length stop using one continuous narration block and
# switch to per-BEAT scripting (see _beat_prompt/plan_beats): each beat's
# line is written and later synthesized to play at that beat's own moment
# in the clip, so narration can never run ahead of or behind the footage.
# Below this, clip length is short enough that a single well-paced block
# stays close enough to in-sync (verified against 35-45s test clips).
BEAT_THRESHOLD = 60.0

# Beat window size. 15s (the original value) was still coarse enough to
# bundle 2-3 separate sub-events from an action-heavy source (e.g. evolve,
# enemy attacks, counter-kick) into ONE beat -- the model wrote one line
# covering all of it, and since that line is read out faster than the
# window is long, the narration raced ahead to describe the LATER sub-event
# while the video was still showing the earlier one (confirmed by pulling
# frames from a real run: "Blaziken walks through" was playing over what
# was still the enemy's attack). 7s keeps most beats to a single sub-event.
BEAT_SPAN = 7.0

# Word budget used ONLY for beat-mode narration -- deliberately higher than
# cfg.words_per_second (2.2, kept conservative for the single-block path,
# see CLAUDE.md). Measured from a real run: Kokoro/edge actually speak at
# ~2.55 words/sec, so budgeting at 2.2 already under-fills every beat by
# ~14% before the model even writes a word -- and on top of that, a
# "max N words" ceiling with no floor pushed the model to write well under
# its own budget. Together that produced 31% silence across a real clip
# (dead air at the tail of nearly every beat) -- the reported "voiceover
# stays silent a lot in between lines" bug. 2.6 fills each beat's window
# close to full at the real observed rate.
BEAT_WORDS_PER_SECOND = 2.6

SYSTEM = (
    "You write voiceover scripts for viral TikTok/Reels/Shorts clips by "
    "ADAPTING the original creator's own dialogue/narration into tight, "
    "punchy English -- you are not writing a separate hype-man reaction "
    "track over it. Stay close to what the source actually says: keep its "
    "specific details, images, and phrasing, just tighten it and give it "
    "energy (translate faithfully if it's in another language -- keep the "
    "content and imagery, don't flatten it into something generic). Add "
    "your own commentary/reaction only where the source itself is thin "
    "(silent action, no narration) or to punch up a transition -- never "
    "replace an already-vivid source line with generic hype filler like "
    "'you won't believe this' or 'wait for it'. "
    "Sound like a real person: conversational, contractions, short punchy "
    "sentences, present tense, specific concrete details (names, numbers, "
    "objects, what's actually said or shown). Punctuation drives the "
    "voice's delivery -- a line that trails off on a comma or has no "
    "ending punctuation gets read completely flat with no emphasis, so "
    "every sentence needs a real ending: a period for a calm beat, an "
    "exclamation mark for a big one (an evolution, a hit landing, a name "
    "being revealed). Don't chain a whole thought into one long "
    "comma-linked clause -- split it into short, separately punctuated "
    "sentences so the delivery actually has punch. "
    "Output ONLY the words to be spoken -- no stage directions, no quotes, "
    "no emoji, no hashtags, no markdown. "
    "ALWAYS write the script in English, even if the source dialogue you are "
    "given is in a different language -- translate/adapt it into English, "
    "never mirror the source language. "
    "The source dialogue is a foreign-language speech-to-text transcript "
    "and sometimes mangles a well-known character/creature/proper name "
    "into a garbled or phonetically-foreign-spelled version (e.g. an ASR "
    "system mishearing a name it doesn't know, or spelling it the way it "
    "sounds in that language rather than its real English spelling). If a "
    "name in the source looks garbled, nonsensical, or like a foreign "
    "transliteration of a name you actually recognize from context (the "
    "video's title, subject, or other names already used), write the real, "
    "correctly-spelled English name instead of repeating the transcript's "
    "error -- never speak a made-up or mis-transcribed name as if it were "
    "correct. "
    "Never narrate ahead of the footage: only react to or adapt dialogue "
    "that has already happened at that point in the script, in the same "
    "order it occurs -- describing a later moment early is the #1 thing "
    "that makes a voiceover feel out of sync with the video."
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
        "footage). Then adapt and comment on events in the SAME order they "
        "occur in the dialogue, ending on the clip's final moment. Translate "
        "and tighten the dialogue's OWN specific details and vivid phrasing "
        "into punchy English rather than replacing them with generic "
        "reaction -- lean on what the source itself actually says/shows, "
        "and only add outside commentary where the source is thin (silent "
        "action) or to punch up a beat. End with a short line reacting to "
        "the final moment that makes the viewer want to comment or rewatch "
        "-- do not add anything after it. "
        "Write the script in ENGLISH regardless of what language the clip's "
        "own dialogue above is in."
    )


def _beat_prompt(cfg: Config, beats: list[dict], context: dict) -> str:
    parts = []
    if context.get("title"):
        head = f"The video is titled: \"{context['title']}\""
        if context.get("uploader"):
            head += f" (channel: {context['uploader']})"
        parts.append(head)
    if context.get("before_text"):
        parts.append(f"Dialogue just BEFORE this clip (for context only):\n"
                     f"{context['before_text']}")
    beat_lines = []
    for i, b in enumerate(beats, 1):
        dur = max(b["end"] - b["start"], 1.0)
        max_w = max(int(dur * BEAT_WORDS_PER_SECOND), 4)
        min_w = max(int(max_w * 0.85), 3)
        beat_lines.append(
            f"BEAT {i} ({b['start']:.0f}s-{b['end']:.0f}s into the clip, "
            f"{min_w}-{max_w} words):\n\"\"\"\n{b['text'] or '(no speech -- action footage)'}\n\"\"\"")
    parts.append("\n\n".join(beat_lines))
    ctx_block = "\n\n".join(parts)
    return (
        f"You are writing the voiceover for a vertical short, split into "
        f"{len(beats)} timed BEATS below -- each beat's narration line "
        "plays DURING that beat's own time window, so it must only "
        "describe what's actually happening in that beat (or an earlier "
        "one) -- never a later beat's events.\n\n"
        f"{ctx_block}\n\n"
        "Output EXACTLY this format, one marker before each beat's line, "
        "nothing else:\n"
        "###BEAT 1###\n<narration for beat 1>\n###BEAT 2###\n<narration for "
        "beat 2>\n...(continue through the last beat, in order)\n\n"
        "Each beat's line MUST land WITHIN that beat's own word range -- "
        "not just under the top of it. Undershooting it is just as wrong "
        "as going over: too few words means dead silence plays for the "
        "back half of that beat before the next one starts, which sounds "
        "just as broken as talking over the wrong footage. If a beat's own "
        "dialogue is thin, don't just write a short line and stop -- add a "
        "touch more reaction/detail to reach the range. The FINAL beat is "
        "the clip's ending -- always describe it, it's the payoff/climax "
        "and must never be cut short for length, even if you have to "
        "compress an earlier, less important beat to make room. For each "
        "beat, translate and tighten THAT beat's own dialogue into punchy "
        "English -- keep its specific details and vivid phrasing rather "
        "than swapping them for generic reaction; only add outside "
        "commentary where a beat has no dialogue (silent action) or to "
        "punch up a transition. Sound like a real person: conversational, "
        "present tense, specific details, no generic filler ('wait for "
        "it', 'you won't believe this'), no stage directions, no markdown, "
        "no emoji, no hashtags. Each beat is voiced as its OWN separate "
        "line -- it MUST end with a period, exclamation mark, or question "
        "mark (never trail off on a comma or nothing), or it plays back "
        "completely flat with no emphasis. Any beat covering a big moment "
        "-- an evolution, a hit landing, a name being revealed, a win -- "
        "MUST end with an exclamation mark, not a period; don't default to "
        "a calm period on a beat that's supposed to hit hard. Don't chain "
        "a beat's whole line into one long comma-linked clause with no "
        "punch at the end -- split it into two short punctuated sentences "
        "if it needs one. Write everything in English regardless of what "
        "language the dialogue above is in."
    )


def _parse_beats(reply: str, n: int) -> list[str] | None:
    parts = [_clean(p) for p in re.split(r"###\s*BEAT\s*\d+\s*###", reply) if p.strip()]
    return parts if len(parts) == n else None


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


def run(cfg: Config, clip: dict, clip_dir, context: dict | None = None,
        beats: list[dict] | None = None) -> str:
    out = clip_dir / "script.txt"
    beats_out = clip_dir / "beats.json"
    already_done = out.exists() and (not beats or beats_out.exists())
    if already_done and not cfg.force:
        log.info("script: %s exists, skipping", out)
        return out.read_text(encoding="utf-8")

    context = context or {}
    duration = clip.get("edited_duration") or (clip["end"] - clip["start"])

    if beats:
        max_tokens = sum(max(int((b["end"] - b["start"]) * BEAT_WORDS_PER_SECOND), 4)
                         for b in beats) * 3
        reply = llm.complete(cfg, _beat_prompt(cfg, beats, context),
                             system=SYSTEM, max_tokens=max_tokens)
        parsed = _parse_beats(reply, len(beats)) if reply else None
        if parsed is None:
            log.info("beat-mode LLM reply unusable -- using per-beat template fallback")
            parsed = [_fallback_script(
                b["text"], max(int((b["end"] - b["start"]) * BEAT_WORDS_PER_SECOND), 6))
                for b in beats]

        for b, text in zip(beats, parsed):
            beat_dur = max(b["end"] - b["start"], 1.0)
            cap = max(int(beat_dur * BEAT_WORDS_PER_SECOND * 1.4), 6)
            words = text.split()
            if len(words) > cap:
                text = " ".join(words[:cap])
            # each beat is its own separate TTS call -- a line with no
            # terminal punctuation gets read back completely flat, which is
            # exactly what caused a real "voiceover sounds dead" report.
            # The prompt now asks for one, but enforce it either way.
            if not re.search(r"[.!?]$", text):
                text += "."
            b["narration"] = text

        script = " ".join(b["narration"] for b in beats)
        clip_dir.mkdir(parents=True, exist_ok=True)
        write_json(beats_out, beats)
        out.write_text(script, encoding="utf-8")
        log.info("beat script (%d beats, %d words): %s...",
                 len(beats), len(script.split()), script[:70])
        return script

    # ---- single continuous block (short clips only) ----
    max_words = int(duration * cfg.words_per_second)
    reply = llm.complete(cfg, _prompt(cfg, clip, duration, context),
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
