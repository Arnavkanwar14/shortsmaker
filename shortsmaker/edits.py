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


def plan_beats(segments: list[dict], clip_start: float, clip_end: float,
              keeps: list[tuple[float, float]] | None, span: float = 15.0) -> list[dict]:
    """Split a clip into ~span-second narration beats on the FINAL (post
    jump-cut) timeline, each carrying the dialogue that actually falls in
    that stretch. This is what lets a long clip's voiceover stay anchored
    to the footage instead of running as one continuous block that drifts
    out of sync (or leaves a silent tail) once the clip runs past ~a
    minute. Beat boundaries prefer not to split a single kept (speech)
    interval, so a beat's text usually comes from one contiguous stretch of
    dialogue -- BUT a keep interval longer than `span` on its own (a source
    with near-continuous narration and few/no silence gaps) still gets
    subdivided into ~span-sized pieces: leaving it as one giant beat means
    that beat's word budget (sized for `span` seconds) can't fill the whole
    window, leaving a long silent gap in the middle of the clip before the
    next beat starts -- a real bug hit on a clip with almost no dead air."""
    dialogue = [(max(s["start"] - clip_start, 0.0),
                min(s["end"] - clip_start, clip_end - clip_start), s["text"])
               for s in segments
               if s["end"] > clip_start and s["start"] < clip_end]

    chunks: list[list[float]] = []   # [orig_start, orig_end, ed_start, ed_end]
    cum = 0.0

    def add_piece(o_s: float, o_e: float) -> None:
        nonlocal cum
        d = o_e - o_s
        if chunks and (cum + d) - chunks[-1][2] <= span:
            chunks[-1][1] = o_e
            chunks[-1][3] = cum + d
        else:
            chunks.append([o_s, o_e, cum, cum + d])
        cum += d

    if keeps:
        for ks, ke in keeps:
            dur = ke - ks
            if dur <= span:
                add_piece(ks, ke)
                continue
            n = max(round(dur / span), 1)
            piece = dur / n
            t = ks
            for i in range(n):
                t_end = ke if i == n - 1 else t + piece
                add_piece(t, t_end)
                t = t_end
    else:
        dur = clip_end - clip_start
        t = 0.0
        while t < dur:
            t_end = min(t + span, dur)
            add_piece(t, t_end)
            t = t_end

    # Each dialogue segment goes to exactly ONE beat -- the one containing
    # its midpoint. The old any-overlap rule put a segment spanning a beat
    # boundary into BOTH beats' text, and the model then dutifully narrated
    # the same source line twice ("It attacks Corphish... It attacks
    # Corphish..."), which was the real cause of a reported
    # sentences-repeat bug (and the resulting overlong beats got trimmed
    # mid-clause, causing the sentences-cut-off half of the same report).
    beats = []
    for orig_s, orig_e, ed_s, ed_e in chunks:
        text = " ".join(t for s, e, t in dialogue
                        if orig_s <= (s + e) / 2 < orig_e).strip()
        beats.append({"start": round(ed_s, 2), "end": round(ed_e, 2), "text": text})

    # a short trailing sliver can't carry its own beat -- fold it back in
    if len(beats) >= 2 and beats[-1]["end"] - beats[-1]["start"] < span * 0.4:
        last = beats.pop()
        beats[-1]["end"] = last["end"]
        beats[-1]["text"] = (beats[-1]["text"] + " " + last["text"]).strip()
    return beats
