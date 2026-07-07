# PLAN: AI B-roll insertion

## Handoff block (paste-first for a fresh session)

shortsmaker (this repo) turns a video into vertical shorts: whisper
transcript → heuristic+Groq highlight selection → optional AI voiceover →
ffmpeg assembly with kinetic captions, snappy-cuts, and dynamic face crop.
This plan adds **AI B-roll**: 2–4 second stock-footage cutaways overlaid at
moments the LLM tags, while audio and captions continue underneath — the
main remaining feature gap vs Opus Clip/Submagic.

Chosen approach: **Groq tags keywords inside the existing virality call
(zero extra API usage) + Pexels free API supplies portrait stock + ffmpeg
overlay before caption burn.** Rejected: generative video B-roll (this
machine is CPU-only — minutes per second, not viable) and source-video
self-cutaways as the primary (a podcast has no "growth chart" shot; kept
only as a fallback when Pexels returns nothing).

Fixed constraints (do not relitigate): free-first — no paid APIs; Groq
stays at ~1 highlights call per run + 1 script call per clip; Windows,
ffmpeg resolved via `shortsmaker.util.ffmpeg_exe()`; every stage must
degrade gracefully (a missing PEXELS_API_KEY or a failed download must
never fail a clip). Read CLAUDE.md gotchas before touching ffmpeg filters.

Prerequisite the owner must do once: create a free key at
https://www.pexels.com/api/ and add `PEXELS_API_KEY=...` to `.env`
(gitignored; `.env.example` gets the placeholder).

Status: not started. Stages below in order; update this file as they land.

---

## Stage 1 — Walking skeleton: one hardcoded cutaway, visible in a real clip

**Endpoint you can SEE:** an existing `final.mp4` from `runs/` re-rendered
with 3 seconds of Pexels stock overlaid mid-clip, captions still on top.

- **Step 1.1 — Pexels fetch module.**
  - Goal: `shortsmaker/broll.py` with `fetch_stock(keyword, cache_dir) -> Path | None`.
  - Where: new file `shortsmaker/broll.py`; key read via a `pexels_api_key()`
    helper in `config.py` (same pattern as `groq_api_key`).
  - How: GET `https://api.pexels.com/videos/search` with header
    `Authorization: <key>`, params `query`, `orientation=portrait`,
    `per_page=3`. Pick the first `video_files` entry with `file_type`
    "video/mp4" and height ≥ 1280. Download to
    `~/.cache/shortsmaker/broll/<slug(keyword)>.mp4`; return the cached file
    on repeat calls without hitting the API.
  - Verify: `python -c "from shortsmaker.broll import fetch_stock; print(fetch_stock('city timelapse', ...))"`
    prints a path; the file plays. Run again — second call returns instantly
    (cache hit, no HTTP).
  - Fence: no pipeline changes, no LLM changes, no UI. Returns `None` on any
    error (missing key, HTTP failure, no results) — never raises.

- **Step 1.2 — Overlay filter proven on a real clip.**
  - Goal: a function `overlay_args(broll_path, at_s, dur_s)` producing the
    ffmpeg filter fragment, tested end-to-end on an existing final.mp4.
  - Where: `shortsmaker/broll.py`; test via a scratch script, not the pipeline.
  - How: broll input scaled/cropped to 1080x1920
    (`scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920`),
    trimmed to `dur_s`, PTS-shifted to `at_s`, overlaid with
    `overlay=enable='between(t,AT,AT+DUR)'`. Critical order note for the
    later integration: in assemble's filter chain the overlay must go
    AFTER crop/scale but BEFORE the `ass=` caption burn, so captions render
    on top of the b-roll.
  - Verify: run the scratch script against
    `runs/<any>/clips/<any>/final.mp4`; open the output; the cutaway appears
    at the right second, audio uninterrupted, captions visible over it.
  - Fence: don't refactor assemble.py yet; the scratch test IS the deliverable.

## Stage 2 — The LLM tags b-roll moments (still one API call)

**Endpoint you can SEE:** `manifest.json` clips carry
`"broll": [{"at": 12.5, "duration": 3, "keyword": "growth chart"}]`.

- **Step 2.1 — Extend the virality rubric response.**
  - Goal: each graded candidate may include up to 2 b-roll suggestions.
  - Where: `shortsmaker/stages/highlights.py` (`llm_virality` prompt + the
    `meta()` parser), `pipeline.py` (copy into the clip entry like
    `metadata`).
  - How: add to the prompt: for each candidate also return
    `"broll": [{"at": <seconds relative to CLIP start>, "keyword": "2-3
    word GENERIC visual concept (city skyline, money counting, gaming
    setup) — something a stock library will actually have"}]` — max 2, only
    where a cutaway genuinely helps; empty list is a good answer. Raise
    `max_tokens` to 2000 (tripwire R3). Parse defensively like `meta()`:
    clamp `at` into [3, clip_dur-4] (never cover the hook or the ending),
    cap at 2 entries, drop malformed ones.
  - Verify: run a fresh auto-detected job (`--llm-provider groq`) on any
    YouTube URL; `manifest.json` shows broll arrays; timestamps fall inside
    the clips; keywords are generic visual concepts.
  - Fence: no new API calls — same single Groq request. No fetching or
    rendering yet.

## Stage 3 — Pipeline integration end-to-end

**Endpoint you can SEE:** a normal run produces clips with b-roll cutaways;
`--broll off` produces identical output to today.

- **Step 3.1 — Assemble renders the tagged cutaways.**
  - Goal: assemble.py accepts a `broll` list (path + at + duration) and
    chains overlays into the existing filter graph.
  - Where: `stages/assemble.py` (filter graph), `pipeline.py` (fetch stock
    per tag via `broll.fetch_stock`, remap `at` through the snappy-cut map
    with `edits.remap_time` — the tag times are pre-cut source-relative,
    the output timeline is post-cut), `config.py` (`broll: str = "auto"`
    — auto | off; auto = on when key present and clip has tags).
  - How: each broll file is an extra `-i` input; chain
    `[v]overlay...[v2]` fragments before the caption burn. If a fetch
    returned None, skip that tag silently (log one line).
  - Verify: full run with tags from stage 2 → open the clips, cutaways
    show at sensible moments with captions on top; then rerun with
    `--broll off` → byte-similar behavior to current main (no overlay
    inputs in the ffmpeg command).
  - Fence: audio graph untouched. Voiceover mode and caption-own-voice mode
    both work — b-roll is video-only. One failed download never fails the clip.

- **Step 3.2 — UI control + visibility.**
  - Goal: a "B-roll" select (auto/off) on the Create form; clip cards show
    a small "N cutaways" note; `.env.example` documents PEXELS_API_KEY.
  - Where: `static/index.html`, `webui.py` (form param), `.env.example`,
    README (one paragraph + the Pexels attribution-not-required note).
  - Verify: toggle produces/omits cutaways through the web UI; a run
    without PEXELS_API_KEY shows clips normally with a single log line
    ("b-roll skipped: no PEXELS_API_KEY").
  - Fence: no redesign of the form; one row, matching the existing style
    tokens.

## Risks & tripwires

- **R1 — Stock relevance is poor for niche keywords.** Tripwire: in step
  1.1's verify, also fetch for 5 realistic keywords from a real transcript;
  if ≥2 look wrong, tighten the prompt toward *generic* concepts (already
  worded that way) before building stage 3. Fallback: skip low-confidence
  tags (LLM returns empty list) — no b-roll beats wrong b-roll.
- **R2 — Windows ffmpeg filter escaping breaks with extra inputs + ass.**
  Tripwire: stage 1.2 tests the exact final chain on a real clip before any
  pipeline wiring. Fallback: two-pass render (overlay pass to temp file,
  then caption pass) — slower but bulletproof.
- **R3 — Bigger JSON truncates the Groq reply** (rubric now carries grades
  + metadata + broll). Tripwire: stage 2 verify includes a 12-candidate
  video; if `extract_json_array` returns None, raise max_tokens to 2500 or
  move broll tagging into the per-clip script call instead (still no new
  calls). 
- **R4 — Pexels rate limit (200/hr)** — only plausible during cache-cold
  batch runs; cache-by-keyword makes repeats free. No action unless hit.

## Decision log

- 2026-07-08: Pexels over Pixabay (bigger catalog, cleaner API, portrait
  filter). Generative b-roll rejected: CPU-only machine. Self-cutaway
  b-roll rejected as primary: concept mismatch; may return as fallback.
- Overlay-under-captions order fixed by design: viewers read captions
  continuously through cutaways; breaking that reads as a glitch.
