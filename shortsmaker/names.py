"""Deterministic proper-name correction against a known-names list.

Whisper transcribing a non-English source (e.g. Spanish Pokemon videos)
regularly mangles names into ASR nonsense or phonetic misspellings
(Blasiken, Gavite, Kombusken, Electros). The LLM prompt asks the model to
fix those from context, but that's best-effort per run -- this layer makes
it deterministic: any capitalized mid-sentence token that closely
fuzzy-matches exactly one name in `known_names.txt` (project root, one
name per line -- ships with all Pokemon species from PokeAPI, editable for
other domains) is replaced with the real spelling. Applied to the
transcript before script generation AND to the generated narration as a
backstop, so a garbled name never reaches TTS or captions.

Guardrails against false positives (e.g. "seeking" -> "Seaking"):
only tokens whose first letter is uppercase are considered, and never at
a sentence start -- mid-sentence capitalization is a proper-noun signal in
both English narration and Whisper's output, while ordinary words are
only ever capitalized at sentence starts. A close-but-ambiguous match
(two candidates within 0.04 of each other) is left alone.
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

log = logging.getLogger("shortsmaker")

_NAMES: list[str] | None = None      # lowercase slugs, dashes kept
_DISPLAY: dict[str, str] = {}        # slug -> "Title Case" display form

MIN_LEN = 5          # shorter tokens fuzzy-match too promiscuously
CUTOFF = 0.80        # Gavite->gabite is 0.83; Blasiken->blaziken 0.88
AMBIGUITY_GAP = 0.04 # two candidates this close = don't guess

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ']*")
_SENT_END = re.compile(r"[.!?:]\s*$|^\s*$")


def _load() -> list[str]:
    global _NAMES
    if _NAMES is None:
        path = Path(__file__).resolve().parent.parent / "known_names.txt"
        _NAMES = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                slug = line.strip().lower()
                if slug:
                    _NAMES.append(slug)
                    _DISPLAY[slug] = slug.replace("-", " ").title()
    return _NAMES


def correct_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (corrected_text, [(from, to), ...])."""
    names = _load()
    if not names or not text:
        return text, []
    fixes: list[tuple[str, str]] = []
    out = []
    last_end = 0
    for m in _TOKEN.finditer(text):
        out.append(text[last_end:m.start()])
        last_end = m.end()
        w = m.group(0)
        lw = w.lower()
        sentence_start = _SENT_END.search(text[:m.start()]) is not None
        if (len(lw) < MIN_LEN or not w[0].isupper() or sentence_start
                or lw in _DISPLAY):
            out.append(w)
            continue
        cand = difflib.get_close_matches(lw, names, n=2, cutoff=CUTOFF)
        if not cand:
            out.append(w)
            continue
        r1 = difflib.SequenceMatcher(None, lw, cand[0]).ratio()
        if len(cand) > 1:
            r2 = difflib.SequenceMatcher(None, lw, cand[1]).ratio()
            if r1 - r2 < AMBIGUITY_GAP:
                out.append(w)
                continue
        fixed = _DISPLAY[cand[0]]
        fixes.append((w, fixed))
        out.append(fixed)
    out.append(text[last_end:])
    return "".join(out), fixes


def correct_transcript(transcript: dict) -> int:
    """Correct all segment texts in place; returns number of fixes."""
    total = 0
    for seg in transcript.get("segments", []):
        seg["text"], fixes = correct_text(seg["text"])
        total += len(fixes)
        for frm, to in fixes:
            log.info("name fix (transcript): %s -> %s", frm, to)
    return total
