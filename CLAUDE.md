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

## Gotchas (all discovered the hard way)
- edge-tts 7.x needs `boundary="WordBoundary"` or you get no word timestamps.
- mediapipe ≥0.10.35 removed `mp.solutions`; use the Tasks API (models
  auto-download to `~/.cache/shortsmaker/`).
- opencv 5 ships no haarcascade XMLs; they're downloaded on demand.
- ffmpeg runs with `cwd=clip_dir` so the .ass caption filter gets a relative
  path — Windows drive colons break filtergraphs otherwise.
- `runs/` folders are ~1GB each (source video kept for re-runs) — gitignored,
  deletable from the Library tab.
