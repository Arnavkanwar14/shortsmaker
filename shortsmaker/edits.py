"""Snappy-cut planning: remove dead air and filler words from a clip.

Works entirely from whisper word timestamps -- no extra models. Produces
"keep" intervals (relative to clip start) that assemble turns into ffmpeg
select filters, plus a remap for caption timestamps onto the compressed
timeline. The result is the jump-cut pacing short-form viewers expect.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("shortsmaker")

# Only unambiguous fillers -- cutting words like "like"/"so" breaks meaning.
FILLER = re.compile(r"^(um+|uh+|uhm+|erm*|ah+|hmm+|mm+|mhm+)[.,!?]*$", re.I)


def plan_cuts(words: list[dict], clip_dur: float,
              max_gap: float = 0.4, pad: float = 0.15,
              min_keep_ratio: float = 0.55) -> list[tuple[float, float]] | None:
    """Keep intervals [(start, end)] relative to clip start, or None when
    cutting is pointless/unsafe (no speech, or it would remove too much).

    max_gap: silences longer than this get trimmed down to 2*pad.
    pad: breathing room kept around speech on both sides of a cut.
    """
    spoken = [w for w in words if not FILLER.match(w["text"].strip())]
    if len(spoken) < 8:                     # not a speech-driven clip
        return None

    keeps: list[list[float]] = []
    for w in spoken:
        s, e = max(w["start"] - pad, 0.0), min(w["end"] + pad, clip_dur)
        if keeps and s - keeps[-1][1] <= max_gap:
            keeps[-1][1] = max(keeps[-1][1], e)
        else:
            keeps.append([s, e])

    kept = sum(e - s for s, e in keeps)
    if kept >= clip_dur - 0.25:             # nothing worth cutting
        return None
    if kept < clip_dur * min_keep_ratio:    # would gut the clip -- refuse
        log.info("snappy cut skipped: would remove %.0f%% of the clip",
                 (1 - kept / clip_dur) * 100)
        return None
    log.info("snappy cut: %.1fs -> %.1fs (%d cuts)", clip_dur, kept, len(keeps) - 1)
    return [(round(s, 3), round(e, 3)) for s, e in keeps]


def edited_duration(keeps: list[tuple[float, float]]) -> float:
    return round(sum(e - s for s, e in keeps), 3)


def remap_words(words: list[dict], keeps: list[tuple[float, float]]) -> list[dict]:
    """Shift word timestamps onto the compressed timeline; words that fall
    inside removed regions (e.g. cut fillers) are dropped."""
    out = []
    offset = 0.0
    for s, e in keeps:
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            if s <= mid <= e:
                out.append({"start": round(w["start"] - s + offset, 3),
                            "end": round(w["end"] - s + offset, 3),
                            "text": w["text"]})
        offset += e - s
    return out


def select_expr(keeps: list[tuple[float, float]]) -> str:
    """ffmpeg select/aselect expression for the keep intervals."""
    return "+".join(f"between(t,{s},{e})" for s, e in keeps)


def remap_time(t: float, keeps: list[tuple[float, float]]) -> float:
    """Map a source-timeline instant onto the compressed timeline.
    Instants inside removed regions collapse to the following cut point."""
    offset = 0.0
    for s, e in keeps:
        if t < s:
            return round(offset, 3)
        if t <= e:
            return round(offset + (t - s), 3)
        offset += e - s
    return round(offset, 3)
