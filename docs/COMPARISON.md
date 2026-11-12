# shortsmaker vs SupoClip

Compared on 2026-11-13 against `github.com/FujiwaraChoki/supoclip` (branch `main`).

## How to read this

- Every SupoClip claim comes from a file listed under "SupoClip files read".
  Those files were read through a fetch tool that returns a detailed
  summary rather than raw source, so constants and function names are as
  reported by that tool. Anything I could not see directly is marked
  **[uncertain]**.
- Every shortsmaker claim comes from the code in this repo
  (`stages/highlights.py`, `llm.py`, `edits.py`, `pipeline.py`, `config.py`,
  `captions.py`, `webui.py`) plus `CLAUDE.md`. `static/index.html` was only
  skimmed for structure (tabs and function list), and `reframe.py` was not
  opened; its description comes from `CLAUDE.md`.

## SupoClip files read

| File | Used for |
|---|---|
| `README.md` | features, stack, requirements |
| `docs/architecture.md` | pipeline, queue, progress streaming, module map |
| `docs/app-guide.md` | screens, clip card contents, editor, publishing |
| `backend/src/ai.py` | selection prompt, scoring, windows, retries, dedup, top-N |
| `backend/src/clip_duration.py` | length presets |
| `backend/src/clip_openings.py` | opening review (fragment and dangling starts) |
| `backend/src/media/timeline.py` | sentence snapping, pause and filler removal, signal summary |
| `backend/src/media/common.py` | constants |
| `backend/src/media/transcription.py` | transcription providers and transcript formatting |
| `backend/src/media/reframing.py` | vertical reframe |
| `backend/src/media/captions.py` | caption and hook-title rendering |
| directory listings of `backend/src` and `frontend/src/components` | file names only |

Not read: the frontend page source, `task_service.py`, `video_utils.py`,
`clip_cleanup.py`, `speaker_framing.py`, `face_detection.py`. Call sites
between modules are therefore **[uncertain]** wherever noted.

## Side by side

### Clip finding

| | shortsmaker | SupoClip |
|---|---|---|
| Candidate source | Local. Windows grown from every sentence start until `min..max_duration`, plus 5s-step sliding windows for low-speech video (`build_windows`, `build_time_windows`). | The LLM proposes segments directly from the transcript. No local candidate generation. |
| Local signals | Scene cuts (PySceneDetect), RMS energy percentile, energy bursts, spectral-centroid expressiveness, speech-rate deviation, keyword density, edge completeness, cut density. Weighted per content profile: talk, action, funny. | `build_clip_signal_summary` in `timeline.py` scores transcript lines on trigger words, questions, exclamations, short lines, pauses, speaker turns and up to 8 audio peaks, and returns the top 12 as text "for the ranking step". Where it is injected is **[uncertain]**. |
| LLM use | 1 call per run. Grades about 6 to 10 pre-scored candidates on hook, flow, value and an overall viral score (0 to 99), may add 2 new windows, and writes upload metadata in the same call. | 1 call for a source up to 15 min. Longer sources are split into 12-min windows with 2-min overlap, 1 call each (3 at a time), then 1 rerank call. |
| Retries | None. An empty or unparseable reply falls back to heuristics and template metadata. | Up to 3 attempts on transient HTTP errors, plus 2 schema-repair retries per agent. |
| Scoring | Final score = 0.5 x heuristic + 0.5 x `viral/99`. The model supplies the overall number itself. Ungraded windows get a neutral prior of 35. | Four 0 to 25 subscores (hook, engagement, value, shareability). The total is recomputed in code as their sum, overwriting the model's total. Also a `hook_type` label. |
| What the LLM sees | Transcript cut at 9000 characters, then candidate texts cut at 280 characters each. On a long video the model never sees most of the transcript. | Whole transcript, as lines of about 8 words with `[MM:SS - MM:SS]` stamps. |
| Clip length | `duration` plus or minus 15s, minimum 15s, set in `webui.py`. | Presets with a hard range and a narrower ideal range: auto 15 to 60 (ideal 25 to 50), short 15 to 30, medium 30 to 60, long 60 to 120. The ideal maximum leaves room for the sentence extension. |
| Start boundary | Window starts on a transcript segment start, then may be moved up to 1.5s onto a scene cut. Nothing checks that the first words stand alone. | Snaps to a sentence-opening word within 3s. Steps back up to 8s (twice at most) when the opening is weak: agreement words, pronouns, back-references. `clip_openings.py` skips 1 or 2-word fragments and short dangling sentences starting with "but", "because", "that's why", within limits of 15s, 12 words, 5 sentences. If it cannot fix the opening it subtracts 10 from the hook score. |
| End boundary | Window growth stops at the first segment ending in `.!?` past `min_duration`. Time-based windows end wherever the span ends. | Extends up to 5.5s to finish the sentence, adds 0.35s padding without running into the next word, and drops a short unfinished tail (12 words or 4s at most). A length cap backs off to the last sentence end. |
| Grounding | Clamps LLM "new" windows into the duration range. Does not snap them to transcript lines. | Snaps LLM timestamps to transcript line boundaries within 8s, replaces model text with transcript text, rejects segments under 3 words. |
| Overlap and dedup | `pick_non_overlapping` allows two picks to overlap by up to 20% of the candidate's length. | Drops any candidate overlapping a kept one by more than 50% of the shorter range, before the rerank. |
| How many clips | Fixed `num_clips` (default 5), always filled if candidates exist. | `ceil(source_seconds / 300)`, at least 5 when the ceiling allows, capped at 20 or the configured maximum. The prompt says to prefer fewer, stronger segments. |
| Low-speech video | Handled: action profile plus time windows. | Selection is transcript-driven, so it has little to work with. This is my inference, not something a SupoClip file states. |

Evidence that boundaries need work here: in the cached run
`runs\https-www-youtube-com-watch-v-irvngjrfzg-23d649\highlights.json` the
first pick (15.0 to 45.0s) opens mid-sentence with "all of us put together
and they died", and it overlaps the second pick (43.13 to 64.34s) by 1.9s.

### Everything else

| | shortsmaker | SupoClip |
|---|---|---|
| Transcription | faster-whisper, local, GPU-capable, word timestamps, sentence segments rebuilt from words. Known-names repair pass. | AssemblyAI by default (paid, speaker labels, language detection). Local Whisper (default model `base`) and YouTube captions are alternatives. |
| Captions | ASS burned by ffmpeg. 4 presets, 2 positions, up to 4 words or 22 characters per phrase, karaoke sweep, pop-in, keyword emphasis. | ASS burned by ffmpeg. Templates with karaoke, pop, bounce and fade, custom fonts, width measured from font metrics, emoji, power-word emphasis, and a 3 to 9 word hook title shown at the top for the first 4s. |
| Reframe | Saliency-based subject tracking, so it works on gameplay and animation. Pan and punch-in track saved as `reframe.json`, hand-editable in the web editor. | Face-based: YuNet with Haar fallback, sampled at 3 fps, spring-smoothed pan that resets on scene cuts, two-speaker split layout, blurred full-frame fit for shots with no face. |
| Voiceover | LLM script plus edge-tts or Kokoro, beat-synced on long clips. | Not present in the files read. The app guide says its AI editor declines voiceover requests. |
| Pause and filler cuts | Yes (`edits.plan_cuts`). | Yes (`build_clip_keep_ranges`). |
| UI stack | One FastAPI process and one `index.html` (1147 lines), tabs Create and Library, status by polling. | Next.js frontend, FastAPI, ARQ worker, Redis, PostgreSQL, Docker Compose. Progress over Server-Sent Events. |
| UI flow | Create form, step indicator and log, clip cards, per-clip reframe and script editor, YouTube upload panel, Library with delete. | Home composer, task list with cancel, resume and bulk delete, task page with clip cards, full timeline editor, publish sheet with scheduling for YouTube, TikTok and Instagram. |
| Clip card | Preview, score, reason, metadata, upload. | Duration, transcript excerpt, AI reasoning, virality score with breakdown, preview, download, Publish. |
| Cost per run | $0. 1 Groq call for highlights and 1 per clip for a voiceover script. | AssemblyAI is billed per audio hour, plus 1 LLM call for a short source or one per 10 min of video and a rerank for a long one, plus 1 call per clip when B-roll is on. Zero only with Whisper and Ollama. Exact prices **[uncertain]**. |

## What to adopt

All of these keep Groq at 1 highlights call per run. None adds a dependency.

| # | Change | File | Effort | Cost impact |
|---|---|---|---|---|
| 1 | Snap every final pick to sentence boundaries using the word timestamps already in the transcript: rewind the start to a sentence-opening word, extend the end to finish the sentence (about 5s limit), drop a short unfinished tail. Apply it to LLM "new" windows too. | `stages/highlights.py` | M | None |
| 2 | Review the opening: skip 1 or 2-word fragments and short sentences that start with a conjunction or back-reference, and lower the hook score when the opening cannot be fixed. | `stages/highlights.py` | S | None |
| 3 | Replace the rubric with four subscores (hook strength, emotional or narrative peak, standalone coherence, payoff), each 0 to 25, summed in code rather than trusted from the model. Add grounding rules and "fewer, stronger". Keep the one-line reason. | `stages/highlights.py` | S | Same single call |
| 4 | Give the LLM the local evidence. Put a short signal digest next to each candidate (energy, burst, speech rate, question or number hits, on a scene cut) so the one call ranks with it, instead of only blending numbers afterwards. | `stages/highlights.py` | S | About 300 more prompt tokens |
| 5 | Stop truncating blindly. For long videos send each candidate's full text with a sentence of context on either side, in place of the first 9000 characters of transcript. The model then judges every candidate on what it actually says. | `stages/highlights.py` | S | Prompt size stays bounded |
| 6 | No overlap at all between picks, with a small gap, and drop near-duplicates before grading so the 6 to 10 graded slots are not spent on the same moment twice. | `stages/highlights.py` | S | None |
| 7 | Sensible top-N: treat `num_clips` as a maximum, scale it with source length, and drop picks under a score floor rather than padding the list with weak clips. | `stages/highlights.py`, `webui.py` | S | Fewer renders and fewer script calls |
| 8 | Length presets with an ideal range inside a hard range (short, medium, long), so sentence extension has room before the cap. | `config.py`, `webui.py`, `static/index.html` | S | None |

Later, for Phase 3 (UI): a clip card showing score breakdown, reason,
duration and transcript excerpt, and optionally a 4-second hook title
burned at the top of the clip (`captions.py`, `stages/assemble.py`, effort
M, no extra call because the title already comes from the highlights call).

## What not to adopt

- AssemblyAI. It is paid, and local Whisper already gives word timestamps.
- Per-window LLM calls, the rerank call, and retry loops. They break the
  Groq budget rule in `CLAUDE.md`.
- The Next.js, Redis, PostgreSQL and Docker stack, accounts, billing and
  admin. This is a single-user localhost tool.
- Server-Sent Events. Polling one local job is enough.

## What shortsmaker does better and must keep

- Local Whisper transcription with GPU support and the known-names repair.
- A free run: one Groq call for highlights, with a heuristic fallback so a
  run still finishes when the LLM is unavailable.
- Multi-signal local scoring and content profiles, which let it clip
  gameplay and other low-speech video.
- Subject-following reframe that does not depend on faces, with an editable
  track.
- AI voiceover with beat sync, custom scripts, and the pacing fixes recorded
  in `CLAUDE.md`.
- Focus keywords, manual spans, whole-clip mode, and upload metadata written
  in the same single call.
- One process, one double-click, no containers.
