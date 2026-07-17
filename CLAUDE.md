# shortsmaker

Free/open-source Viewmax/Opus-Clip alternative: video in → vertical shorts
with optional AI voiceover, kinetic captions, virality-graded highlight
selection. Python package + FastAPI web UI. Repo: Arnavkanwar14/shortsmaker.

## Run
- `.venv\Scripts\activate`, then `python -m shortsmaker web` → 127.0.0.1:8000
- CLI: `python -m shortsmaker run --input <file-or-url> ...`
- Verify a pipeline change by running on the cached test run in `runs\`
  (delete `clips\` + `highlights.json` inside a run to re-do just the
  cheap stages — download/transcript stay cached).

## Architecture (stage files under shortsmaker/stages/)
ingest → transcribe (faster-whisper) → highlights (heuristics + 1 Groq
rubric call) → script (LLM) → tts (edge-tts) → assemble (ffmpeg). Each
stage writes to `runs/<run-id>/` and skips if its output exists.

## Groq budget rule
Keep LLM usage at ~1 call per run for highlights + 1 per clip for scripts.
Never add per-candidate or retry-loop LLM calls — free tier.

## YouTube upload (shortsmaker/youtube.py)
Free YouTube Data API v3. Owner drops a "Desktop app" OAuth client JSON at
project root as `youtube_client_secret.json` (gitignored); "Connect" runs
`InstalledAppFlow.run_local_server` once, saving `youtube_token.json`.
Upload ≈1600 quota units → ~6/day. Vertical + `#Shorts` in the description
makes it register as a Short. Both json files are gitignored.

## Voiceover pacing/sync (decided after a real reported bug)
`words_per_second` is deliberately 2.2, under natural TTS pace, so the
fit-check in tts.py rarely triggers a speed-up; if it does, `MAX_VO_SPEEDUP`
is capped at 1.10 (was 1.25 -- audibly rushed, the actual cause of a
reported "voiceover talks too fast"). script_gen's prompt explicitly
forbids narrating events out of the transcript's own chronological order
(no foreshadowing a later reveal in the opening hook) -- don't relax either
without re-testing against a multi-beat clip (setup ... late reveal).

script_gen's hard-trim safety net is 1.6x the word budget (was 1.15x) and
trims at a sentence boundary, not a blind word slice -- at 1.15x it kept
chopping off the clip's climax (always last, since the model writes in
order), which is exactly the "script ends early, missing the fight" bug
a real user hit. The prompt now also explicitly tells the model to budget
itself so it reaches the final beat, compressing early filler instead.
Don't tighten the 1.6x cap without re-testing a long multi-beat clip
against the actual climax appearing in the output.

For clips >= script_gen.BEAT_THRESHOLD (60s), narration is no longer one
continuous audio block -- that was the REAL bug behind repeat reports of
"script ends early / describes the evolution too soon / silence at the
end" on ~2min clips: natural TTS pace outruns the footage over a long
clip, so a single block finishes way before the video does. Now
edits.plan_beats() splits the clip into ~15s beats on the POST-CUT
timeline, script_gen writes one line per beat in ONE LLM call (still
respects the Groq budget rule), and tts.py synthesizes+places each beat
at its own timestamp via ffmpeg adelay/amix. A moment literally cannot be
narrated before it's on screen anymore. Short clips (<60s) still use the
old single-block path, verified unaffected. If this regresses, verify
with a synthetic multi-beat 2min transcript (see the scratchpad test
pattern) rather than guessing from the code.

Beat span (script_gen.BEAT_SPAN) is 7s, not the original 15s -- at 15s a
single beat could still bundle 2-3 separate sub-events from an
action-heavy source (evolve, enemy attacks, counter-kick), and since the
model wrote ONE line for the whole window, that line read out faster
than the window and raced ahead to the later sub-event while the video
was still on the earlier one. Confirmed by pulling actual frames from a
real run, not by guessing -- "Blaziken walks through" was playing over a
frame that was still the enemy's attack. 7s keeps most beats to one
sub-event. Don't widen this without re-checking real extracted frames
against the captions at several timestamps, not just word-count math --
the beat-fit math looked fine at 15s too; the bug was in sub-beat
ordering, which only shows up by actually looking at the video.

## Gotchas (all discovered the hard way)
- HF serverless Inference API no longer hosts TTS (routes to PAID partner
  providers) -- abandoned in favor of Kokoro-82M running locally via the
  `kokoro` package. Don't re-attempt the API route.
- GPU whisper works on this machine via pip `nvidia-cublas-cu12` +
  `nvidia-cudnn-cu12` (no CUDA Toolkit needed), but the DLL dirs must be
  prepended to PATH -- `transcribe._register_cuda_dlls()` does it.
  GPU only pays off on `small`+ models (~2.4x); `tiny` is CPU-bound.
- Batched whisper returns coarse ~30s segments; `_sentence_segments()`
  rebuilds sentence granularity from word timestamps -- keep it, the
  highlight windowing depends on it.
- edge-tts 7.x needs `boundary="WordBoundary"` or you get no word timestamps.
- mediapipe ≥0.10.35 removed `mp.solutions`; use the Tasks API (models
  auto-download to `~/.cache/shortsmaker/`).
- opencv 5 ships no haarcascade XMLs; they're downloaded on demand.
- ffmpeg runs with `cwd=clip_dir` so the .ass caption filter gets a relative
  path — Windows drive colons break filtergraphs otherwise.
- `runs/` folders are ~1GB each (source video kept for re-runs) — gitignored,
  deletable from the Library tab.
