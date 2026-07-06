# shortsmaker

Free/open-source pipeline that turns a source video (local file **or** URL)
into 3–5 vertical (9:16) short clips with AI voiceover commentary and
kinetic auto-captions — a no-subscription alternative to credit-based SaaS
tools like Viewmax.

Every component is free: local models where possible, keyless free services
where not. No paid API key is required by default.

## Install

```bash
cd shortsmaker
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on mac/linux)
pip install -r requirements.txt
```

ffmpeg is resolved automatically: `SHORTSMAKER_FFMPEG` env var → system
PATH → the binary bundled with `imageio-ffmpeg`. **Note:** some static
ffmpeg builds lack libass; if caption burning fails the pipeline still
exports the clip without captions and logs a warning — installing a full
ffmpeg build (e.g. `winget install Gyan.FFmpeg`) fixes it.

### LLM for commentary scripts (pick one, all free)

The script/highlight LLM is swappable via `--llm-provider` (default `auto`,
which tries them in this order):

| Provider | Cost | Setup | Notes |
|---|---|---|---|
| `ollama` | free, unlimited, offline | install https://ollama.com, `ollama pull llama3.1:8b` | best privacy, needs decent hardware |
| `groq` | free tier (~14k req/day) | set `GROQ_API_KEY` (console.groq.com) | Llama 3.3 70B, very fast, best quality/effort |
| `gemini` | free tier (~1.5k req/day) | set `GEMINI_API_KEY` (aistudio.google.com) | Gemini 2.0 Flash |
| `none` | free | nothing | heuristic highlights + template scripts |

Without any of these the pipeline still completes using heuristics and a
template script — an LLM just makes the commentary genuinely good.

**Where to put API keys:** copy `.env.example` to `.env` in the project root
and paste your key(s) there (the file is gitignored). Environment variables
work too and take precedence: `setx GROQ_API_KEY "gsk_..."` on Windows
(then open a new terminal), or `export GROQ_API_KEY=...` on mac/linux.

Other optional extras:
- **mediapipe** for face-aware cropping (`pip install mediapipe`); otherwise
  center crop.
- **piper-tts** for fully offline voiceover (`--tts-engine piper --piper-model
  voice.onnx`); default is edge-tts (free online Microsoft voices, no key).
- **iopaint** for Stage 7 watermark/caption removal (`pip install iopaint`).

## Usage

### Web UI

```bash
python -m shortsmaker web            # then open http://127.0.0.1:8000
```

Upload a video or paste a URL, pick clip count/length/voice/caption style
and LLM provider, watch the pipeline log live, then preview and download
the finished clips in the browser.

### CLI

```bash
# from a URL
python -m shortsmaker run --input "https://www.youtube.com/watch?v=..." \
    --num-clips 5 --duration 45 --voice en-US-GuyNeural --style kinetic

# from a local file, no LLM, plain captions
python -m shortsmaker run --input talk.mp4 --llm-provider none --style plain

# pick a specific LLM
python -m shortsmaker run --input talk.mp4 --llm-provider gemini

# list voices
python -m shortsmaker voices --lang en
```

## How it works (stages)

Each stage writes intermediate files to `runs/<run-id>/`, so any stage can
be re-run independently (finished outputs are skipped unless `--force`):

| # | Stage | Tool | Outputs |
|---|-------|------|---------|
| 1 | Ingest | yt-dlp + ffmpeg | `normalized.mp4` |
| 2 | Transcribe | faster-whisper (local) | `transcript.json` (word-level) |
| 3 | Highlights | PySceneDetect + librosa heuristics + optional LLM | `highlights.json` |
| 4 | Script | Ollama / Groq / Gemini free tiers / template fallback | `clips/clip_NN/script.txt` |
| 5 | Voiceover | edge-tts (word timestamps from boundary events) or piper | `voiceover.mp3`, `vo_words.json` |
| 6 | Assemble | ffmpeg + mediapipe face crop + generated `.ass` | `captions.ass`, `final.mp4` |
| 7 | Cleanup (opt-in `--clean true`) | LaMa via iopaint | `cleaned.mp4` |

Highlight scoring combines audio energy percentile, keyword/emotion density
(questions, superlatives, numbers, laughter), speech-rate deviation,
sentence completeness, and scene-cut alignment; an optional LLM pass boosts
windows it independently selects. One failing clip never kills the batch —
its manifest entry is marked `failed` and the rest proceed.

Each run ends with `manifest.json`: which moments were chosen and why, the
scripts used, per-clip status, and a **SaaS cost-equivalent ledger** showing
roughly what a credit-based tool would have charged for each stage.

## Realistic expectations

- CPU-only works but is slow (whisper + any local LLM benefit from a GPU).
- Expect to tune the voice, caption style, and prompts to taste; occasional
  rough cuts may need a manual trim.
- Only run this on video you own or have the rights to repurpose.
