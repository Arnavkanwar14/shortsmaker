"""STAGE 3 -- FIND THE BEST MOMENTS.

Combines several free signals (no single one is trusted alone):
  - PySceneDetect shot boundaries (windows that start on a cut feel cleaner)
  - audio RMS energy percentiles via librosa (loud/energetic passages)
  - speech-rate deviation (fast excited talking or dramatic slow-downs)
  - keyword/emotion density (questions, superlatives, numbers, laughter)
  - sentence completeness at the window edges
The best distinct candidates are then graded in ONE free LLM call on a
four-part rubric (hook / peak / coherence / payoff). Final picks are
snapped to sentence boundaries, never overlap, and are ranked best-first.

Output: highlights.json -- ranked [{rank, start, end, score, reason,
signals, virality}]
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np

from .. import llm
from ..config import Config, load_channel
from ..util import read_json, write_json

log = logging.getLogger("shortsmaker")

# Settings that change which moments get picked. highlights.json used to be
# cached purely by "does the file exist", so changing --focus (or
# content_type, num_clips, duration...) and re-running the same video
# silently kept the OLD clip selection -- this fingerprints the inputs that
# actually affect selection so a change forces a fresh pick.
HIGHLIGHTS_SETTINGS_KEYS = [
    "focus", "content_type", "num_clips", "min_duration", "max_duration",
    "llm_provider", "use_llm_highlights", "scene_threshold",
    "ideal_min_duration", "ideal_max_duration",
]

# Bump when the selection LOGIC changes (rubric, snapping, dedup...) so a
# cached highlights.json from older logic is recomputed, not reused.
HIGHLIGHTS_LOGIC_VERSION = 2


def highlights_signature(cfg: Config) -> str:
    d = {k: getattr(cfg, k) for k in HIGHLIGHTS_SETTINGS_KEYS}
    d["_v"] = HIGHLIGHTS_LOGIC_VERSION
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

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


# Openings that lean on something said earlier. English + Spanish (most
# sources here are Spanish). A bare "so"/"y" is left out on purpose: it
# usually opens a real claim.
DANGLING = {"but", "because", "and", "or", "which", "although", "though",
            "however", "then", "also", "plus", "meanwhile", "anyway",
            "pero", "porque", "entonces", "aunque", "sino", "ademas",
            "además", "tambien", "también", "luego"}
DANGLING_PHRASES = ("that's why", "thats why", "which means", "as i said",
                    "like i said", "as i mentioned", "por eso", "es decir",
                    "o sea", "como dije")
# Weak (not wrong) openers: fine if the previous sentence can be pulled in.
WEAK_OPENERS = {"yeah", "yes", "yep", "okay", "ok", "exactly", "right",
                "well", "like", "he", "she", "they", "them", "him", "her",
                "si", "sí", "claro", "bueno", "pues", "vale"}
FILLER_TOKENS = {"um", "uh", "uhh", "uhm", "umm", "hmm", "eh", "este"}
SECOND_PERSON = re.compile(r"\b(you|your|you're|tu|tú|te|tus)\b", re.I)


def _tokens(text: str) -> list[str]:
    toks = [re.sub(r"[^\w']", "", t.lower()) for t in text.split()]
    return [t for t in toks if t and t not in FILLER_TOKENS]


def is_dangling(text: str) -> bool:
    toks = _tokens(text)
    if not toks:
        return False
    return toks[0] in DANGLING or " ".join(toks[:6]).startswith(DANGLING_PHRASES)


def is_weak_opening(text: str) -> bool:
    toks = _tokens(text)
    return bool(toks) and (toks[0] in WEAK_OPENERS or is_dangling(text))


def hook_score(first_text: str) -> float:
    """Cheap local read of the opening line: questions, numbers, strong
    words and direct address grab; an opening that depends on earlier
    context does not."""
    sent = re.split(r"(?<=[.!?…])\s+", first_text.strip())[0]
    n = len(sent.split())
    if not n:
        return 0.0
    s = 0.2
    s += 0.35 if "?" in sent else 0.0
    s += 0.2 if NUMBERS.search(sent) else 0.0
    s += 0.2 if SUPERLATIVES.search(sent) else 0.0
    s += 0.15 if SECOND_PERSON.search(sent) else 0.0
    s += 0.1 if n <= 14 else 0.0
    s -= 0.4 if is_weak_opening(sent) else 0.0
    return min(max(s, 0.0), 1.0)


def completeness_score(first_text: str, last_text: str) -> float:
    score = 0.0
    if first_text[:1].isupper() or first_text[:1] in "\"'":
        score += 0.5
    if SENTENCE_END.search(last_text):
        score += 0.5
    return score


def cut_density_score(cuts: list[float], start: float, end: float) -> float:
    """Fast cutting / rapid shot changes signal action. ~4 cuts/10s -> 1.0."""
    n = sum(1 for c in cuts if start <= c <= end)
    per10 = n / max((end - start) / 10.0, 0.1)
    return min(per10 / 4.0, 1.0)


def energy_burst_score(times, rms, start: float, end: float) -> float:
    """Sudden loudness spikes (explosions, shouting, crowd pops) vs the
    track's own baseline. Peak z-score inside the window, capped."""
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return 0.0
    mu, sigma = float(rms.mean()), float(rms.std()) or 1e-9
    z = (float(rms[mask].max()) - mu) / sigma
    return min(max(z, 0.0) / 4.0, 1.0)


def audio_expressiveness(wav: Path):
    """Return (times, centroid) -- spectral centroid track for the whole
    audio. A fully local, free signal (no LLM): its local variance is a
    fast proxy for animated vocal delivery (rising/falling emphasis,
    exclamations, surprise) vs a flat monotone read, without the cost of
    full pitch tracking (librosa.pyin is much slower on long audio)."""
    import librosa
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    hop = 512
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop)
    return times, centroid


def expressiveness_score(times, centroid, start: float, end: float) -> float:
    """Coefficient of variation of spectral centroid inside the window,
    relative to the track's own variability. High = animated/excited
    delivery; low = flat/monotone."""
    mask = (times >= start) & (times <= end)
    if mask.sum() < 4:
        return 0.0
    local_mean = float(centroid[mask].mean()) or 1e-9
    local_cv = float(centroid[mask].std()) / local_mean
    track_mean = float(centroid.mean()) or 1e-9
    track_cv = float(centroid.std()) / track_mean or 1e-9
    return min(local_cv / (track_cv * 1.5), 1.0)


# Signal weights per content profile. "talk" is the original podcast/vlog
# tuning; "action" (gaming, sports) trusts the audio+editing rhythm over
# dialogue; "funny" hunts laughter and spikes.
# "hook" took half of completeness's old weight in talk/funny: boundaries
# are now repaired by snap_clip() after picking, so edge completeness
# matters less as a ranking signal and the opening line matters more.
PROFILES = {
    "talk":   {"energy": .25, "keywords": .25, "speech_rate_dev": .10,
               "completeness": .10, "scene_aligned": .10, "hook": .10,
               "cut_density": .00, "energy_burst": .00, "expressiveness": .10},
    "action": {"energy": .25, "keywords": .05, "speech_rate_dev": .00,
               "completeness": .05, "scene_aligned": .10, "hook": .00,
               "cut_density": .25, "energy_burst": .30, "expressiveness": .00},
    "funny":  {"energy": .15, "keywords": .30, "speech_rate_dev": .10,
               "completeness": .05, "scene_aligned": .05, "hook": .05,
               "cut_density": .00, "energy_burst": .20, "expressiveness": .10},
}

# plain-language names for the one-line "why this clip" when no LLM graded it
SIGNAL_WORDS = {
    "energy": "loud, energetic delivery", "keywords": "punchy wording",
    "speech_rate_dev": "a change of pace", "completeness": "a complete thought",
    "scene_aligned": "a clean cut at the start", "hook": "a strong opening line",
    "cut_density": "fast cutting", "energy_burst": "a sudden loud moment",
    "expressiveness": "animated delivery", "focus": "a match for your focus",
}


def detect_content_type(segments: list[dict], duration: float) -> str:
    """Cheap auto-detect: sparse speech -> action; lots of laughter -> funny."""
    words = sum(len(s["text"].split()) for s in segments)
    wpm = words / max(duration / 60.0, 0.1)
    laughs = sum(len(LAUGHTER.findall(s["text"])) for s in segments)
    if wpm < 50:
        return "action"
    if laughs >= 3:
        return "funny"
    return "talk"


# ----------------------------------------------------------- windowing
def _window_texts(segments: list[dict], start: float, end: float) -> tuple[str, str, str]:
    """(joined text, first segment text, last segment text) inside a span."""
    inside = [s for s in segments if s["start"] < end and s["end"] > start]
    if not inside:
        return "", "", ""
    return " ".join(s["text"] for s in inside), inside[0]["text"], inside[-1]["text"]


def _ideal_range(cfg: Config) -> tuple[float, float]:
    """Range candidate windows are built in. It sits inside the hard
    min..max range so snapping a boundary to a sentence has room to move."""
    lo = cfg.ideal_min_duration or cfg.min_duration
    hi = cfg.ideal_max_duration or max(cfg.max_duration - 5, cfg.min_duration + 5)
    return float(lo), float(hi)


def build_windows(segments: list[dict], cfg: Config) -> list[dict]:
    """Grow candidate windows from each segment start until min..max duration."""
    windows = []
    n = len(segments)
    lo, hi = _ideal_range(cfg)
    for i in range(n):
        start = segments[i]["start"]
        j = i
        while j < n and segments[j]["end"] - start < lo:
            j += 1
        while j < n and segments[j]["end"] - start <= hi:
            windows.append({
                "start": start, "end": segments[j]["end"],
                "text": " ".join(s["text"] for s in segments[i:j + 1]),
                "first_text": segments[i]["text"], "last_text": segments[j]["text"],
            })
            # prefer stopping at sentence ends: keep only sentence-final
            # extensions after the first valid one
            if SENTENCE_END.search(segments[j]["text"]):
                break
            j += 1
    return windows


def build_time_windows(segments: list[dict], duration: float, cfg: Config) -> list[dict]:
    """Transcript-independent sliding windows for low-speech content
    (gameplay, sports, montages) where dialogue can't anchor the clips."""
    windows = []
    lo, hi = _ideal_range(cfg)
    span = min(max(float(cfg.duration or (lo + hi) / 2), lo), hi)
    t = 0.0
    while t + lo <= duration:
        end = min(t + span, duration)
        text, first, last = _window_texts(segments, t, end)
        windows.append({"start": t, "end": end, "text": text,
                        "first_text": first, "last_text": last})
        t += 5.0
    return windows


def snap_to_scene(t: float, cuts: list[float], tol: float = 1.5) -> tuple[float, bool]:
    for c in cuts:
        if abs(c - t) <= tol:
            return c, True
    return t, False


def focus_score(text: str, focus: str) -> float:
    """Fraction of the user's focus terms present in a window's text."""
    terms = [t for t in re.split(r"[,;/]|\s+", focus.lower()) if len(t) > 2]
    if not terms:
        return 0.0
    tl = text.lower()
    return sum(1 for t in terms if t in tl) / len(terms)


def score_windows(windows, cuts, times, rms, cfg: Config, profile: str,
                  centroid=None) -> list[dict]:
    weights = PROFILES[profile]
    speech_rates = [len(w["text"].split()) / max(w["end"] - w["start"], 1)
                    for w in windows]
    mean_rate = float(np.mean(speech_rates)) if speech_rates else 1.0
    std_rate = float(np.std(speech_rates)) or 1.0

    scored = []
    for w, rate in zip(windows, speech_rates):
        start, on_cut = snap_to_scene(w["start"], cuts)
        signals = {
            "energy": window_energy_pct(times, rms, w["start"], w["end"]),
            "keywords": keyword_score(w["text"]),
            "speech_rate_dev": min(abs(rate - mean_rate) / std_rate / 2.0, 1.0),
            "completeness": completeness_score(w["first_text"], w["last_text"]),
            "scene_aligned": 1.0 if on_cut else 0.0,
            "hook": hook_score(w["first_text"]),
            "cut_density": cut_density_score(cuts, w["start"], w["end"]),
            "energy_burst": energy_burst_score(times, rms, w["start"], w["end"]),
            "expressiveness": (expressiveness_score(times, centroid, w["start"], w["end"])
                               if centroid is not None else 0.0),
        }
        score = sum(weights[k] * v for k, v in signals.items())
        shown = {k: round(v, 3) for k, v in signals.items() if weights[k]}
        if cfg.focus:
            # user asked for specific moments: a strong extra signal on top
            # of whatever the content profile scores
            fs = focus_score(w["text"], cfg.focus)
            score += 0.45 * fs
            shown["focus"] = round(fs, 3)
        scored.append({
            "start": round(start, 2), "end": round(w["end"], 2),
            "score": round(score, 4),
            "heuristic": round(score, 4),
            "signals": shown,
            "text": w["text"],
            "reason": _heuristic_reason(shown, weights),
        })
    return scored


def _heuristic_reason(shown: dict, weights: dict) -> str:
    """One readable line from the two signals that contributed most."""
    top = sorted(shown, key=lambda k: weights.get(k, 0.45) * shown[k], reverse=True)
    top = [SIGNAL_WORDS[k] for k in top[:2] if shown[k] >= 0.4]
    return ("Picked for " + " and ".join(top) if top
            else "Best available moment by audio and editing signals") + "."


def _overlap(a: dict, b: dict) -> float:
    return min(a["end"], b["end"]) - max(a["start"], b["start"])


def dedupe_candidates(cands: list[dict], ratio: float = 0.5) -> list[dict]:
    """Best-first, drop any candidate that overlaps a kept one by more than
    `ratio` of the shorter of the two. Windows are grown from every sentence
    start, so without this the handful of slots the LLM grades get spent on
    near-copies of the same moment."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda w: w["score"], reverse=True):
        if all(_overlap(c, k) <= ratio * min(c["end"] - c["start"],
                                             k["end"] - k["start"]) for k in kept):
            kept.append(c)
    return kept


# ------------------------------------------------- sentence boundaries
START_SNAP = 3.0        # prefer a sentence start this close to the raw start
START_REACH = 8.0       # ...and accept one this far when none is closer
END_EXTEND = 5.5        # finish the sentence if it ends within this
END_BACKOFF = 6.0       # else drop an unfinished tail no longer than this
END_PAD = 0.35          # breath after the last word
LEAD_IN = 0.1           # breath before the first word
OPENING_MAX_SKIP = 15.0 # how far an opening fix may move the start
HOOK_PENALTY = 10       # off the hook subscore (of 25) for a bad opening


def build_sentences(segments: list[dict]) -> list[dict]:
    """Sentence spans [{start, end, text}] from transcript segments. A
    segment that was only split by the transcriber's word cap (no terminal
    punctuation, no pause) is merged back into the sentence it belongs to."""
    sents: list[dict] = []
    for s in segments:
        if (sents and not SENTENCE_END.search(sents[-1]["text"])
                and s["start"] - sents[-1]["end"] < 0.8):
            sents[-1]["end"] = s["end"]
            sents[-1]["text"] += " " + s["text"]
        else:
            sents.append({"start": s["start"], "end": s["end"], "text": s["text"]})
    return sents


def _inside(sents: list[dict], starts: list[float], t: float,
            eps: float = 0.05) -> int | None:
    """Index of the sentence being spoken at instant t, else None."""
    import bisect
    i = bisect.bisect_right(starts, t - eps) - 1
    if i >= 0 and sents[i]["start"] + eps < t < sents[i]["end"] - eps:
        return i
    return None


def snap_clip(c: dict, sents: list[dict], starts: list[float], cuts: list[float],
              lo: float, hi: float) -> dict:
    """Return a copy of candidate `c` whose start and end sit on sentence
    boundaries, with the opening reviewed. Never leaves the lo..hi range:
    a move that would is skipped. Adds c["edges"], a note per change, and
    c["opening_issue"] when the opening still leans on missing context."""
    import bisect
    n = len(sents)
    start, end = c["start"], c["end"]
    notes: list[str] = []

    # ---- start: never begin inside a sentence
    k = _inside(sents, starts, start)          # sentence the clip opens on
    moved = False
    if k is not None:
        back = start - sents[k]["start"]
        fwd = sents[k + 1]["start"] - start if k + 1 < n else float("inf")
        if back <= START_SNAP and end - sents[k]["start"] <= hi:
            moved = True
        elif fwd <= START_SNAP and end - sents[k + 1]["start"] >= lo:
            k, moved = k + 1, True
        elif back <= START_REACH and end - sents[k]["start"] <= hi:
            moved = True
        elif fwd <= START_REACH and end - sents[k + 1]["start"] >= lo:
            k, moved = k + 1, True
        else:
            k = None
            notes.append("starts mid-sentence (no boundary close enough)")
    else:
        i = bisect.bisect_left(starts, start - 0.05)
        # speech begins right away -> the opening line is the hook; a clip
        # that opens on seconds of pure action has a visual hook instead
        k = i if i < n and sents[i]["start"] - start <= 1.5 else None

    # ---- opening review (only when the clip opens on speech)
    issue = ""
    if k is not None:
        k0 = k
        for _ in range(2):      # pull in the previous sentence for context
            if not (k > 0 and is_weak_opening(sents[k]["text"])
                    and sents[k]["start"] - sents[k - 1]["start"] <= START_REACH
                    and sents[k]["start"] - sents[k - 1]["end"] <= 2.0
                    and end - sents[k - 1]["start"] <= hi):
                break
            k -= 1

        def can_skip_to(j: int) -> bool:
            return (j < n and sents[j]["start"] < end
                    and sents[j]["start"] - sents[k0]["start"] <= OPENING_MAX_SKIP
                    and end - sents[j]["start"] >= lo)

        # 2+ leading throwaway fragments ("Right. Okay.") -- one alone is
        # left in place: "Here's why." is a hook, not banter
        j = k
        while j < n and len(_tokens(sents[j]["text"])) <= 2:
            j += 1
        if j - k >= 2 and can_skip_to(j):
            k = j
        # a short sentence that only continues an earlier thought
        if (is_dangling(sents[k]["text"]) and len(_tokens(sents[k]["text"])) <= 6
                and can_skip_to(k + 1) and not is_weak_opening(sents[k + 1]["text"])):
            k += 1
        if is_dangling(sents[k]["text"]):
            issue = "opens mid-thought"
        if k != k0:
            moved = True
            notes.append("opening moved to a self-contained sentence")
        elif moved:
            notes.append("start snapped to sentence start")

    if moved and k is not None:
        first = sents[k]["start"]
        prev_end = sents[k - 1]["end"] if k > 0 else 0.0
        start = max(first - LEAD_IN, prev_end, 0.0)
        # a shot change in the silence just before the first word is the
        # cleanest possible place to cut in
        on_cut = [x for x in cuts if max(prev_end, first - 1.5) <= x <= first]
        if on_cut:
            start = on_cut[-1]

    # ---- end: never stop inside a sentence
    j = _inside(sents, starts, end)
    if j is not None:
        nxt = sents[j + 1]["start"] if j + 1 < n else float("inf")
        full = min(sents[j]["end"] + END_PAD, max(nxt - 0.03, sents[j]["end"]))
        prev_end = sents[j - 1]["end"] if j > 0 else None
        if sents[j]["end"] - end <= END_EXTEND and full - start <= hi:
            end = full
            notes.append("end extended to finish the sentence")
        elif (prev_end is not None and prev_end > start
              and end - sents[j]["start"] <= END_BACKOFF
              and prev_end + END_PAD - start >= lo):
            end = min(prev_end + END_PAD, sents[j]["start"] - 0.03)
            notes.append("unfinished last sentence dropped")
        else:
            notes.append("ends mid-sentence (no boundary close enough)")

    out = dict(c, start=round(start, 2), end=round(end, 2))
    if notes:
        out["edges"] = notes
    if issue:
        out["opening_issue"] = issue
    return out


def select_clips(scored: list[dict], segments: list[dict], cuts: list[float],
                 cfg: Config, duration: float, llm_graded: bool,
                 profile: str = "talk") -> list[dict]:
    """Final picks: best-first, each snapped to sentence boundaries, no
    overlap at all, ranked by score. `num_clips` is a ceiling, not a quota:
    short sources and weak candidates produce fewer clips."""
    sents = build_sentences(segments)
    starts = [s["start"] for s in sents]
    lo, hi = float(cfg.min_duration), float(cfg.max_duration)
    # a source can only hold so many separate clips worth watching
    k = max(1, min(cfg.num_clips, int(duration // (2 * lo))))

    picked: list[dict] = []
    for c in sorted(scored, key=lambda w: w["score"], reverse=True):
        if len(picked) >= k:
            break
        if llm_graded and "virality" not in c:
            continue        # the model saw the best of every region already
        w = snap_clip(c, sents, starts, cuts, lo, hi)
        if any(_overlap(w, p) > 0 for p in picked):
            continue
        w["text"], _, _ = _window_texts(segments, w["start"], w["end"])
        if w.get("opening_issue"):
            v = w.get("virality")
            if v and "peak" in v:
                v = w["virality"] = dict(v, hook=max(v["hook"] - HOOK_PENALTY, 0))
                v["viral"] = v["hook"] + v["peak"] + v["coherence"] + v["payoff"]
                w["score"] = _blend(w.get("heuristic", w["score"]), v["viral"], profile)
            else:
                w["score"] = round(w["score"] - 0.05, 4)
            w["reason"] = w["reason"].rstrip(".") + f" ({w['opening_issue']})."
        picked.append(w)

    picked.sort(key=lambda w: w["score"], reverse=True)
    if llm_graded:
        floor = [w for w in picked
                 if w["virality"]["viral"] >= MIN_VIRAL.get(profile, 40)]
    else:
        floor = [w for w in picked if w["score"] >= 0.6 * picked[0]["score"]] if picked else []
    if picked and len(floor) < len(picked):
        log.info("dropped %d weak pick(s) below the quality floor",
                 len(picked) - len(floor or picked[:1]))
        picked = floor or picked[:1]
    for rank, w in enumerate(picked, 1):
        w["rank"] = rank
    return picked


# Graded picks under this rubric total (of 100) are not rendered. Action
# video is mostly wordless, and the model only reads words, so its total
# runs low there and is trusted less (see _blend).
MIN_VIRAL = {"talk": 40, "funny": 40, "action": 35}
LLM_WEIGHT = {"talk": 0.7, "funny": 0.7, "action": 0.5}


def _blend(heuristic: float, viral: int, profile: str = "talk") -> float:
    """The model judges the content, the heuristics the delivery. On
    speech-driven video the model's rubric total carries most of the
    weight; on action video the audio/editing signals count equally."""
    w = LLM_WEIGHT.get(profile, 0.7)
    return round((1 - w) * heuristic + w * viral / 100, 4)


# ------------------------------------------------------------ LLM pass
CONTENT_HINTS = {
    "talk": "strong hook, complete thought, emotional or surprising content",
    "action": "the most intense action: kills, clutch plays, crashes, saves, "
              "big reactions -- editing rhythm and excitement matter more than "
              "complete sentences",
    "funny": "the funniest moments: laughter, absurd reactions, comedic "
             "timing, things people would tag a friend under",
}


def _as_list(v) -> list[str]:
    """Groq sometimes returns a requested JSON array as a comma-separated
    string instead -- iterating a raw string yields one entry per
    character, so coerce explicitly rather than trusting the type."""
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def _dedupe_cap(items: list[str], max_n: int, max_chars: int) -> list[str]:
    """Order-preserving dedupe (case-insensitive), capped by count and by
    total comma-joined length (YouTube's tag budget)."""
    out, seen, total = [], set(), 0
    for it in items:
        it = it.strip()
        key = it.lower()
        if not it or key in seen:
            continue
        if len(out) >= max_n or total + len(it) + 2 > max_chars:
            break
        out.append(it)
        seen.add(key)
        total += len(it) + 2
    return out


def finalize_metadata(g: dict, channel: dict) -> dict:
    """Turn one LLM grade into upload-ready metadata, applying universal SEO
    rules plus the optional channel profile.

    Returns {title, description, hashtags, tags}. The description is the full
    text to paste into YouTube (hook + story + related-searches + CTA +
    hashtag line); hashtags/tags are also returned separately for the API.
    """
    title = str(g.get("title", "")).strip().lstrip("#")[:100]
    body = str(g.get("description", "")).strip()
    related = str(g.get("related", "")).strip()

    # hashtags: channel-fixed (branded consistency) override the LLM's;
    # otherwise the LLM's 3-tier set. Always <=5, always includes a shorts
    # tag so the upload registers as a Short.
    if channel.get("hashtags"):
        hashtags = [str(h).lstrip("#").replace(" ", "") for h in channel["hashtags"] if h]
    else:
        hashtags = [t.lstrip("#").replace(" ", "") for t in _as_list(g.get("hashtags"))]
    hashtags = hashtags[:5]
    if not any(h.lower() == "shorts" for h in hashtags):
        hashtags = (["Shorts"] + hashtags)[:5]

    # tags: always-include channel tags first (consistency), then the LLM's
    # subject-specific tags; deduped, <=15 and <500 chars (YouTube limits).
    fixed = [str(t) for t in (channel.get("tags") or [])]
    subj = [t.lower() for t in _as_list(g.get("tags"))]
    tags = _dedupe_cap(fixed + subj, 15, 480)

    # assemble the full description deterministically so the rules (CTA,
    # related line, hashtag block) are guaranteed, not left to the model.
    cta = channel.get("cta") or (
        f"Subscribe to {channel['name']} for more." if channel.get("name")
        else "Subscribe for more.")
    parts = [body]
    if related:
        parts.append("Related searches: " + related)
    parts.append(cta)
    if hashtags:
        parts.append(" ".join("#" + h for h in hashtags))
    description = "\n\n".join(p for p in parts if p)[:1200]

    return {"title": title, "description": description,
            "hashtags": hashtags, "tags": tags}


def _diverse_pool(candidates: list[dict], n: int, duration: float) -> list[dict]:
    """Up to n candidates spread across the WHOLE video timeline, instead
    of pure top-score -- which can cluster entirely inside one loud or
    keyword-dense stretch and never even show the LLM a candidate from
    the rest of a long video. Best-scoring candidate per time-bucket
    first, then top-score overall fills any remaining slots."""
    if duration <= 0 or not candidates:
        return sorted(candidates, key=lambda c: c["score"], reverse=True)[:n]
    n_buckets = max(n, 1)
    bucket_w = duration / n_buckets
    best_per_bucket: dict[int, dict] = {}
    for c in candidates:
        b = min(int(((c["start"] + c["end"]) / 2) / bucket_w), n_buckets - 1)
        if b not in best_per_bucket or c["score"] > best_per_bucket[b]["score"]:
            best_per_bucket[b] = c
    diverse = sorted(best_per_bucket.values(), key=lambda c: c["score"], reverse=True)
    seen = {(c["start"], c["end"]) for c in diverse}
    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if len(diverse) >= n:
            break
        key = (c["start"], c["end"])
        if key not in seen:
            diverse.append(c)
            seen.add(key)
    return diverse[:n]


def _template_metadata(c: dict, meta: dict | None, channel: dict) -> dict:
    """Local, no-LLM upload metadata built directly from the clip's own
    transcript excerpt and the source video's title. Used whenever the
    grading call is unavailable (rate-limited, no key, provider down) so
    clips still get a usable title/description/tags instead of nothing --
    mirrors the existing template fallback for voiceover scripts. Routed
    through finalize_metadata() so the CTA/hashtag/related assembly and
    channel-profile branding stay identical to the LLM path."""
    text = (c.get("text") or "").strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    video_title = ((meta or {}).get("title") or "").strip()
    title = video_title or (sentences[0] if sentences else "Watch this moment")
    body = " ".join(sentences[:2]).strip() or title
    fake_grade = {"title": title, "description": body, "related": "",
                 "hashtags": [], "tags": []}
    return finalize_metadata(fake_grade, channel)


SUBSCORES = ("hook", "peak", "coherence", "payoff")
ACTION_NOTE = (
    "This is action footage with little speech, so few or no words is "
    "normal and must not lower a score by itself: lean on the signals. "
    "High burst/loud/cuts means something is happening on screen -- treat "
    "that as the hook and the peak. Coherence = one continuous sequence, "
    "payoff = the sequence resolves rather than cutting away mid-moment. ")
HOOK_TYPES = {"question", "statement", "statistic", "story", "contrast", "none"}
FULL_TRANSCRIPT_CHARS = 9000    # under this the model reads the whole thing
MAX_POOL = 24                   # most candidates graded on a long video
LISTING_CHARS = 13000           # ...as long as their text fits in this


def _clip_excerpt(text: str, limit: int = 1000) -> str:
    """Whole clip text, or its head + tail when long: the opening and the
    ending are what hook and payoff get judged on."""
    text = text or "(no speech)"
    if len(text) <= limit:
        return text
    return text[:int(limit * 0.65)] + " [...] " + text[-int(limit * 0.35):]


def _signal_digest(c: dict) -> str:
    names = {"energy": "loud", "energy_burst": "burst", "cut_density": "cuts",
             "expressiveness": "expressive", "speech_rate_dev": "pace-shift",
             "keywords": "keywords", "hook": "hook-words", "focus": "focus"}
    sig = c.get("signals", {})
    return ", ".join(f"{names[k]} {sig[k]:.2f}" for k in names if k in sig) or "none"


def _plain(text: str) -> str:
    """Models like typographic hyphens and quotes; keep the one-line reason
    plain ASCII punctuation so it renders the same everywhere."""
    for a, b in (("‑", "-"), ("‐", "-"), ("–", "-"), ("—", " - "),
                 ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(a, b)
    return " ".join(text.split())


def _subscores(g: dict) -> dict | None:
    """The four rubric subscores as ints 0-25, or None if the model left
    them out. A model that answers on a 0-100 scale is rescaled."""
    try:
        vals = [float(g[k]) for k in SUBSCORES]
    except (KeyError, TypeError, ValueError):
        return None
    if max(vals) > 25:
        vals = [v / 4 for v in vals]
    return {k: int(round(min(max(v, 0), 25))) for k, v in zip(SUBSCORES, vals)}


def llm_virality(cfg: Config, segments: list[dict], candidates: list[dict],
                 duration: float, profile: str,
                 meta: dict | None = None) -> tuple[list[dict], bool]:
    """Grade the best distinct candidates on a four-part rubric -- hook /
    peak / coherence / payoff, 0-25 each, total summed HERE not by the
    model -- in ONE LLM call per run (free-tier friendly). The model sees
    each candidate's local audio/editing measurements, may suggest windows
    the heuristics missed (short videos only), and writes the upload
    metadata. Returns (candidates, graded?)."""
    candidates = dedupe_candidates(candidates)
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    transcript_block = "\n".join(lines)
    whole = len(transcript_block) <= FULL_TRANSCRIPT_CHARS
    # Groq's free tier allows 8000 tokens a minute, prompt + reply. Scores
    # are cheap (~50 tokens a candidate) but upload metadata is not (~220),
    # so every candidate gets scored and only the best few get metadata.
    # A short video sends its whole transcript and ~10 candidates; a long
    # one drops the transcript and spends that room on up to MAX_POOL
    # candidates spread across the timeline.
    n_meta = cfg.num_clips + 1
    n_grade = min(max(cfg.num_clips * 2, 6), 10) if whole else MAX_POOL
    top = _diverse_pool(candidates, n_grade, duration)

    focus_matches: list[dict] = []
    if cfg.focus:
        # a literal focus match (e.g. "boss fights") can score low on the
        # heuristics (audio energy, cut density...), which never involve
        # topic -- without this, it might never reach the LLM to be graded
        # at all. Guarantee real keyword hits are in the pool it sees.
        by_focus = sorted(candidates, key=lambda c: focus_score(c["text"], cfg.focus),
                          reverse=True)
        focus_matches = [c for c in by_focus if focus_score(c["text"], cfg.focus) > 0][:6]
        seen, merged = set(), []
        for c in focus_matches + top:
            key = (c["start"], c["end"])
            if key not in seen:
                seen.add(key)
                merged.append(c)
        top = merged

    top_keys = {(c["start"], c["end"]) for c in top}
    rest = [c for c in candidates if (c["start"], c["end"]) not in top_keys]

    new_window_cap = (4 if cfg.focus else 2) if whole else 0
    sents = build_sentences(segments)

    def listing_entry(i: int, c: dict) -> str:
        head = (f'{i}: [{c["start"]:.0f}s-{c["end"]:.0f}s, '
                f'{c["end"] - c["start"]:.0f}s] signals: {_signal_digest(c)}')
        body = f'   clip: "{_clip_excerpt(c["text"], 1000 if whole else 700)}"'
        if whole:
            return f"{head}\n{body}"
        # long video: the transcript will not fit, so give each candidate
        # the sentence on either side instead -- enough to judge whether
        # it stands alone
        before = [s["text"] for s in sents if s["end"] <= c["start"] + 0.2][-1:]
        after = [s["text"] for s in sents if s["start"] >= c["end"] - 0.2][:1]
        return (f"{head}\n"
                f'   said just before (NOT in the clip): "{(before or ["(nothing)"])[0][-150:]}"\n'
                f"{body}\n"
                f'   said just after (NOT in the clip): "{(after or ["(nothing)"])[0][:150]}"')

    # keep the listing inside the prompt budget: drop from the END of the
    # pool (lowest priority) until it fits, and move those back to `rest`
    entries = [listing_entry(i, c) for i, c in enumerate(top)]
    while len(top) > 6 and sum(len(e) for e in entries) > LISTING_CHARS:
        entries.pop()
        top.pop()
    top_keys = {(c["start"], c["end"]) for c in top}
    rest = [c for c in candidates if (c["start"], c["end"]) not in top_keys]
    listing = "\n".join(entries)
    context = (f"Transcript (timestamped):\n{transcript_block or '(almost no speech in this video)'}\n\n"
               if whole else
               "The full transcript is too long to include; each candidate "
               "comes with the sentence spoken just before and after it.\n\n")

    focus_line = (
        f"HARD CONSTRAINT, not a preference: the user wants clips "
        f"specifically about \"{cfg.focus}\". Any candidate that is NOT "
        f"genuinely about \"{cfg.focus}\" must get subscores totalling <=30 "
        f"no matter how engaging it otherwise looks -- only clips actually "
        f"showing/discussing \"{cfg.focus}\" may total above 50. Any NEW "
        f"windows you add must be moments matching \"{cfg.focus}\" the "
        f"heuristics missed.\n\n"
    ) if cfg.focus else ""
    src_line = ""
    if meta and meta.get("title"):
        src_line = (f"Source video title: \"{meta['title']}\""
                    + (f" (channel: {meta['uploader']})" if meta.get("uploader") else "")
                    + " -- use its names/topic as the primary search keywords.\n")
    new_line = (
        f"You may also add up to {new_window_cap} NEW windows the list missed, "
        f"{cfg.min_duration}-{cfg.max_duration}s long, starting on the first word "
        "of a sentence and ending on the last word of one, using timestamps "
        "from the transcript. " if new_window_cap else "")
    new_example = (
        ', {"id": "new", "start": 120.0, "end": 165.0, "hook": 17, "peak": 15, '
        '"coherence": 19, "payoff": 14, "hook_type": "story", "reason": "one line", '
        '"title": "...", "description": "...", "related": "...", '
        '"hashtags": ["..."], "tags": ["..."]}' if new_window_cap else "")
    prompt = (
        f"A {duration:.0f}-second video. {src_line}{context}"
        f"Candidate clips to grade. \"signals\" are local measurements on a 0-1 "
        "scale: loud = loudness vs the rest of the video, burst = a sudden "
        "loud spike, cuts = shot-change rate, expressive = vocal variation, "
        "pace-shift = unusual speaking speed, keywords/hook-words = questions, "
        "numbers and strong words. They tell you about delivery and editing "
        "energy that text cannot show: use them as evidence for hook and peak, "
        "but strong signals never rescue empty content.\n"
        f"{listing}\n\n{focus_line}"
        "Judge each candidate as a short a stranger sees with ZERO context "
        f"(this video's style: {CONTENT_HINTS[profile]}). Use only what the "
        "lines actually say; never assume context that is not in the clip. "
        f"{ACTION_NOTE if profile == 'action' else ''}"
        "Give four integer subscores, each 0-25:\n"
        "- hook: does the first sentence / first 3 seconds stop the scroll? "
        "20+ = a bold claim, question, number, conflict or stakes right away; "
        "10-15 = clear but ordinary; under 8 = greeting, filler, or it needs "
        "what was said before.\n"
        "- peak: the emotional or narrative high point inside the clip -- "
        "surprise, conflict, a reveal, laughter, real stakes. 20+ only when "
        "something actually happens or lands.\n"
        "- coherence: does it make complete sense alone? One idea, starting "
        "at the start of a thought, no unexplained 'he/that/it' pointing "
        "outside the clip.\n"
        "- payoff: does it END on a landed point -- answer, punchline, result, "
        "takeaway? Under 8 if it trails off or stops before the resolution.\n"
        "Do NOT give a total: it is computed from your subscores. Be strict "
        "and spread the scores -- most clips total under 60, intros, sponsor "
        "reads, recaps and sign-offs total under 30, and near-identical "
        "scores for every candidate are a wrong answer.\n"
        "- hook_type: question | statement | statistic | story | contrast | none.\n"
        "- reason: ONE line, at most 18 words, naming the concrete thing that "
        "makes a viewer watch or share it (or the flaw that sinks it).\n"
        "Write reason/title/description/hashtags/tags in ENGLISH regardless of "
        "what language the transcript is in.\n"
        f"Score ALL {len(top)} candidates: the array must hold one object "
        f"for every id from 0 to {len(top) - 1}. Most objects are short -- "
        "just id, the four subscores, hook_type and reason. For your "
        f"{n_meta} highest-scoring candidates ONLY, add these upload "
        "metadata fields to the same object:\n"
        "- title: <=70 chars, NO hashtags, NO emoji, NO quotes. Front-load the "
        "main subject/keyword people would search, then open a curiosity gap "
        "and do NOT reveal the payoff (make them click to find out). Use an "
        "impossible-sounding statement, an incomplete reveal, a direct 'you', "
        "or an irony/contradiction.\n"
        "- description: a hook line matching the title's energy, then tell it "
        "in order -- setup, then the key fact, then the twist. Name the "
        "subject in the first sentence (SEO). No hashtags and no call-to-"
        "action in the text (both are added automatically).\n"
        "- related: 4-6 comma-separated search phrases people type to find this.\n"
        "- hashtags: exactly 5, no # symbol, ordered specific->niche->broad "
        "(main subject, then subject+facts/lore, then niche, then broad like "
        "shorts/viral/fyp).\n"
        "- tags: 6-10 lowercase keyword tags, subject-specific first.\n"
        f"{new_line}Respond with ONLY a JSON array, one object per candidate: "
        '[{"id": 0, "hook": 18, "peak": 12, "coherence": 20, "payoff": 9, '
        '"hook_type": "question", "reason": "one line", "title": "...", '
        '"description": "...", "related": "phrase a, phrase b", '
        '"hashtags": ["subject", "..."], "tags": ["subject", "..."]}, '
        '{"id": 1, "hook": 7, "peak": 9, "coherence": 14, "payoff": 6, '
        '"hook_type": "none", "reason": "one line"}'
        f'{new_example}]'
    )
    reply = llm.complete(cfg, prompt,
                         max_tokens=3500,
                         temperature=0.2,     # grading, not writing: keep it repeatable
                         system="You are a senior short-form video editor who "
                                "predicts which clips go viral. Your job is "
                                "ranking, not creative rewriting. You are "
                                "strict: most clips total under 60.")
    grades = llm.extract_json_array(reply)
    channel = load_channel()      # per-channel branded/fixed constants
    if channel:
        log.info("metadata: applying channel profile %r", channel.get("name", "?"))

    if not grades:
        log.info("LLM virality pass unavailable/unparseable (%d chars); "
                 "using heuristics + template metadata only", len(reply or ""))
        for c in candidates:
            c["metadata"] = _template_metadata(c, meta, channel)
        return candidates, False

    def virality(g: dict, sub: dict) -> dict:
        ht = str(g.get("hook_type", "none")).strip().lower()
        return dict(sub, viral=sum(sub.values()),
                    hook_type=ht if ht in HOOK_TYPES else "none",
                    reason=_plain(str(g.get("reason", ""))))

    graded = [dict(c) for c in top]
    added = 0
    for g in grades:
        if not isinstance(g, dict):
            continue
        sub = _subscores(g)
        if sub is None:
            continue
        try:
            if g.get("id") == "new":
                ps, pe = float(g["start"]), float(g["end"])
                ps = max(0.0, min(ps, duration - cfg.min_duration))
                pe = min(max(pe, ps + cfg.min_duration),
                         min(ps + cfg.max_duration, duration))
                if pe - ps < cfg.min_duration * 0.8 or added >= new_window_cap:
                    continue
                text, _, _ = _window_texts(segments, ps, pe)
                v = virality(g, sub)
                graded.append({
                    "start": round(ps, 2), "end": round(pe, 2),
                    "score": _blend(v["viral"] / 100, v["viral"], profile),
                    "heuristic": v["viral"] / 100, "signals": {},
                    "text": text, "reason": v["reason"] or "Found by the model.",
                    "virality": v,
                    **({"metadata": finalize_metadata(g, channel)} if g.get("title") else {}),
                })
                added += 1
                continue
            c = graded[int(g["id"])]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        v = c["virality"] = virality(g, sub)
        if g.get("title"):
            c["metadata"] = finalize_metadata(g, channel)
        c["score"] = _blend(c["heuristic"], v["viral"], profile)
        c["reason"] = v["reason"] or c["reason"]
    n_graded = sum(1 for c in graded if "virality" in c)
    for c in sorted(graded, key=lambda c: c["score"], reverse=True):
        v = c.get("virality")
        if v:
            log.info("  graded %6.1f-%6.1fs total=%2d hook=%2d peak=%2d coh=%2d payoff=%2d  %s",
                     c["start"], c["end"], v["viral"], v["hook"], v["peak"],
                     v["coherence"], v["payoff"], v["reason"][:70])
    log.info("LLM virality pass: %d candidates graded, %d new windows "
             "(1 API call, %d prompt chars)", n_graded - added, added, len(prompt))
    if not n_graded:
        for c in candidates:
            c["metadata"] = _template_metadata(c, meta, channel)
        return candidates, False
    for c in graded + rest:
        # anything the model did not write metadata for still gets a usable
        # title/description, built locally from its own transcript
        if "metadata" not in c:
            c["metadata"] = _template_metadata(c, meta, channel)
    return graded + rest, True


# ---------------------------------------------------------------- main
def run(cfg: Config, video: Path, transcript: dict,
        meta: dict | None = None) -> list[dict]:
    out = cfg.run_dir / "highlights.json"
    sig = highlights_signature(cfg)
    sig_file = cfg.run_dir / "highlights_sig.txt"
    stale = not sig_file.is_file() or sig_file.read_text(encoding="utf-8").strip() != sig
    if out.exists() and not cfg.force and not stale:
        log.info("highlights: highlights.json exists, skipping")
        return read_json(out)
    if stale and out.exists() and not cfg.force:
        log.info("highlights: focus/content-type/clip-count settings changed "
                 "-- recomputing instead of reusing the old selection")

    segments = [s for s in transcript["segments"] if s["text"].strip()]

    # scene detection is ~10s/min of video -- cache it per run so setting
    # tweaks and re-runs don't pay it again
    scenes_file = cfg.run_dir / "scenes.json"
    if scenes_file.exists() and not cfg.force:
        cuts = read_json(scenes_file)
        log.info("scene detection: %d shots (cached)", len(cuts))
    else:
        cuts = detect_scenes(video, cfg.scene_threshold)
        write_json(scenes_file, cuts)
    wav = cfg.run_dir / "audio_16k.wav"
    times, rms = audio_energy(wav)
    duration = float(times[-1]) if len(times) else 0.0
    try:
        # same wav/sr/hop as audio_energy -> its own times array lines up
        # frame-for-frame with `times` above, so score_windows can index
        # both rms and centroid with the single `times` array
        _, centroid = audio_expressiveness(wav)
    except Exception as e:
        log.info("expressiveness signal unavailable (%s); scoring without it", str(e)[:120])
        centroid = None

    profile = cfg.content_type
    if profile not in PROFILES:
        profile = detect_content_type(segments, duration)
        log.info("content type auto-detected: %s", profile)
    else:
        log.info("content type: %s (user-set)", profile)

    windows = build_windows(segments, cfg)
    # low-speech content (gameplay, sports, montages) can't be anchored to
    # dialogue -- add transcript-independent sliding windows as well
    if profile == "action" or len(windows) < cfg.num_clips * 3:
        windows += build_time_windows(segments, duration, cfg)
    log.info("built %d candidate windows", len(windows))
    if not windows:
        raise RuntimeError("video too short to build any candidate window")

    scored = score_windows(windows, cuts, times, rms, cfg, profile, centroid)
    graded = False
    if cfg.use_llm_highlights:
        scored, graded = llm_virality(cfg, segments, scored, duration, profile, meta)

    picked = select_clips(scored, segments, cuts, cfg, duration, graded, profile)

    if cfg.focus:
        # be honest about whether the picks actually deliver the requested
        # content -- a graded viral score >50 means the LLM judged it a
        # genuine match (the prompt hard-gates non-matches below 50);
        # ungraded picks fall back to the keyword heuristic.
        def _focus_matched(c: dict) -> bool:
            v = c.get("virality")
            if v and "viral" in v:
                return v["viral"] > 50
            return focus_score(c.get("text", ""), cfg.focus) > 0

        for w in picked:
            w["focus_matched"] = _focus_matched(w)
        if not any(w["focus_matched"] for w in picked):
            log.warning("focus %r: none of the picked clips genuinely match -- "
                       "the video may not contain this content; showing the "
                       "best available picks instead", cfg.focus)

    for w in picked:
        log.info("pick #%d %.1f-%.1fs score=%.2f (%s)", w["rank"], w["start"],
                 w["end"], w["score"], w["reason"][:60])
    write_json(out, picked)
    sig_file.write_text(sig, encoding="utf-8")
    return picked
