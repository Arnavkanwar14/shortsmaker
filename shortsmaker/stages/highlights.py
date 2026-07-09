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
]


def highlights_signature(cfg: Config) -> str:
    d = {k: getattr(cfg, k) for k in HIGHLIGHTS_SETTINGS_KEYS}
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


# Signal weights per content profile. "talk" is the original podcast/vlog
# tuning; "action" (gaming, sports) trusts the audio+editing rhythm over
# dialogue; "funny" hunts laughter and spikes.
PROFILES = {
    "talk":   {"energy": .30, "keywords": .30, "speech_rate_dev": .10,
               "completeness": .20, "scene_aligned": .10,
               "cut_density": .00, "energy_burst": .00},
    "action": {"energy": .25, "keywords": .05, "speech_rate_dev": .00,
               "completeness": .05, "scene_aligned": .10,
               "cut_density": .25, "energy_burst": .30},
    "funny":  {"energy": .15, "keywords": .35, "speech_rate_dev": .10,
               "completeness": .10, "scene_aligned": .05,
               "cut_density": .00, "energy_burst": .25},
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
    span = float(cfg.duration or (cfg.min_duration + cfg.max_duration) / 2)
    t = 0.0
    while t + cfg.min_duration <= duration:
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


def score_windows(windows, cuts, times, rms, cfg: Config, profile: str) -> list[dict]:
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
            "cut_density": cut_density_score(cuts, w["start"], w["end"]),
            "energy_burst": energy_burst_score(times, rms, w["start"], w["end"]),
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
            "signals": shown,
            "text": w["text"],
            "reason": f"heuristic[{profile}]: " + (", ".join(
                k for k, v in shown.items() if v >= 0.6) or "mixed signals"),
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


def llm_virality(cfg: Config, segments: list[dict], candidates: list[dict],
                 duration: float, profile: str, meta: dict | None = None) -> list[dict]:
    """Grade the top candidates on an Opus-style rubric -- Hook / Flow /
    Value, 0-99 -- in ONE LLM call per run (free-tier friendly). The model
    may also suggest up to 2 windows the heuristics missed, and writes
    SEO-optimized YouTube Shorts upload metadata for each candidate."""
    # full upload metadata per candidate is token-heavy, so grade fewer:
    # ~2x the requested clip count is plenty to pick from without blowing
    # the single-call output budget (risk R3 in PLAN.md)
    n_grade = min(max(cfg.num_clips * 2, 6), 10)
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)

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
        for c in focus_matches + ranked:
            key = (c["start"], c["end"])
            if key not in seen:
                seen.add(key)
                merged.append(c)
        ranked = merged
        n_grade = min(n_grade + len(focus_matches), 14)

    top = ranked[:n_grade]
    rest = ranked[n_grade:]
    new_window_cap = 4 if cfg.focus else 2

    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    transcript_block = "\n".join(lines) or "(almost no speech in this video)"
    if len(transcript_block) > 9000:             # stay cheap on free tiers
        transcript_block = transcript_block[:9000] + "\n[...transcript trimmed...]"
    listing = "\n".join(
        f'{i}: [{c["start"]:.0f}s-{c["end"]:.0f}s] "{(c["text"] or "(no speech)")[:280]}"'
        for i, c in enumerate(top))

    focus_line = (
        f"HARD CONSTRAINT, not a preference: the user wants clips "
        f"specifically about \"{cfg.focus}\". Any candidate that is NOT "
        f"genuinely about \"{cfg.focus}\" must be scored viral <=30 no "
        f"matter how engaging it otherwise looks -- only clips actually "
        f"showing/discussing \"{cfg.focus}\" may score above 50. Any NEW "
        f"windows you add must be moments matching \"{cfg.focus}\" the "
        f"heuristics missed.\n\n"
    ) if cfg.focus else ""
    src_line = ""
    if meta and meta.get("title"):
        src_line = (f"Source video title: \"{meta['title']}\""
                    + (f" (channel: {meta['uploader']})" if meta.get("uploader") else "")
                    + " -- use its names/topic as the primary search keywords.\n")
    prompt = (
        f"A {duration:.0f}-second video. {src_line}Transcript (timestamped):\n"
        f"{transcript_block}\n\n"
        f"Candidate clips to grade:\n{listing}\n\n{focus_line}"
        f"Grade EACH candidate as a standalone viral short ({CONTENT_HINTS[profile]}). "
        "Score 0-99 on: hook (do the first seconds grab attention?), "
        "flow (complete thought, no mid-idea chop?), value (takeaway, emotion, "
        "or aha-moment?), and overall viral score.\n"
        "Also write upload metadata per candidate:\n"
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
        f"You may also add up to {new_window_cap} NEW windows the list missed, using "
        f'{cfg.min_duration}-{cfg.max_duration}s spans. Respond with ONLY a JSON array: '
        '[{"id": 0, "hook": 70, "flow": 80, "value": 60, "viral": 71, '
        '"reason": "one line", "title": "...", "description": "...", '
        '"related": "phrase a, phrase b", "hashtags": ["subject", "..."], '
        '"tags": ["subject", "..."]}, '
        '{"id": "new", "start": 120.0, "end": 165.0, "viral": 75, '
        '"reason": "one line", "title": "...", "description": "...", '
        '"related": "...", "hashtags": ["..."], "tags": ["..."]}]'
    )
    reply = llm.complete(cfg, prompt, max_tokens=4000 + 400 * len(focus_matches),
                         system="You are a short-form video editor who predicts "
                                "which clips go viral. You are strict: most "
                                "clips score under 60.")
    grades = llm.extract_json_array(reply)
    if not grades:
        log.info("LLM virality pass unavailable/unparseable (%d chars); "
                 "using heuristics only", len(reply or ""))
        return candidates

    channel = load_channel()      # per-channel branded/fixed constants
    if channel:
        log.info("metadata: applying channel profile %r", channel.get("name", "?"))

    def meta(g: dict) -> dict:
        return finalize_metadata(g, channel)

    graded = [dict(c) for c in top]
    added = 0
    for g in grades:
        try:
            if g.get("id") == "new":
                ps, pe = float(g["start"]), float(g["end"])
                ps = max(0.0, min(ps, duration - cfg.min_duration))
                pe = min(max(pe, ps + cfg.min_duration),
                         min(ps + cfg.max_duration, duration))
                if pe - ps < cfg.min_duration * 0.8 or added >= new_window_cap:
                    continue
                text, _, _ = _window_texts(segments, ps, pe)
                viral = int(g.get("viral", 60))
                graded.append({
                    "start": round(ps, 2), "end": round(pe, 2),
                    "score": round(0.9 * viral / 99, 4), "signals": {},
                    "text": text, "reason": f"LLM found: {g.get('reason', '')}",
                    "virality": {"viral": viral, "reason": g.get("reason", "")},
                    "metadata": meta(g),
                })
                added += 1
                continue
            c = graded[int(g["id"])]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        viral = int(g.get("viral", 0))
        c["virality"] = {
            "hook": int(g.get("hook", 0)), "flow": int(g.get("flow", 0)),
            "value": int(g.get("value", 0)), "viral": viral,
            "reason": g.get("reason", ""),
        }
        c["metadata"] = meta(g)
        # blend: heuristics know the audio/editing, the LLM knows the content
        c["score"] = round(0.5 * c["score"] + 0.5 * viral / 99, 4)
        c["reason"] = f"viral {viral}: {g.get('reason', '')}"
    # ungraded windows must compete on the same scale as graded ones --
    # blend them with a low neutral prior, otherwise a raw heuristic 0.7
    # beats a graded viral-80 (blended ~0.65) and ungraded clips win picks
    prior = 35 / 99
    for c in graded + rest:
        if "virality" not in c:
            c["score"] = round(0.5 * c["score"] + 0.5 * prior, 4)
    n_graded = sum(1 for c in graded if "virality" in c)
    log.info("LLM virality pass: %d candidates graded, %d new windows (1 API call)",
             n_graded, added)
    return graded + rest


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

    scored = score_windows(windows, cuts, times, rms, cfg, profile)
    if cfg.use_llm_highlights:
        scored = llm_virality(cfg, segments, scored, duration, profile, meta)

    picked = pick_non_overlapping(scored, cfg.num_clips)
    for w in picked:
        log.info("pick %.1f-%.1fs score=%.2f (%s)", w["start"], w["end"],
                 w["score"], w["reason"][:60])
    write_json(out, picked)
    sig_file.write_text(sig, encoding="utf-8")
    return picked
