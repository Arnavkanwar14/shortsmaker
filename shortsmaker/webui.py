"""Web UI for shortsmaker: upload a video or paste a URL, watch the
pipeline log live, preview and download the finished clips.

Run:  python -m shortsmaker web  [--port 8000]
"""
from __future__ import annotations

import collections
import logging
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .pipeline import derive_run_id, run as run_pipeline
from .util import setup_logging

log = logging.getLogger("shortsmaker")

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
UPLOADS_DIR = RUNS_DIR / "_uploads"

app = FastAPI(title="shortsmaker")

# ---------------------------------------------------------------- jobs
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()          # one pipeline at a time (CPU-heavy)


class _JobLogHandler(logging.Handler):
    def __init__(self, buffer: collections.deque):
        super().__init__(logging.INFO)
        self.buffer = buffer
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.buffer.append(self.format(record))


def _worker(job_id: str, cfg: Config) -> None:
    job = JOBS[job_id]
    handler = _JobLogHandler(job["log"])
    logging.getLogger("shortsmaker").addHandler(handler)
    def _progress(stage: str, detail: str) -> None:
        job["stage"], job["stage_detail"] = stage, detail

    try:
        with JOB_LOCK:
            job["status"] = "running"
            manifest = run_pipeline(cfg, progress=_progress)
        ok = sum(1 for c in manifest["clips"] if c["status"] == "ok")
        job["manifest"] = manifest
        job["status"] = "done" if ok else "failed"
    except Exception as e:
        log.error("job %s failed: %s", job_id, e)
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        logging.getLogger("shortsmaker").removeHandler(handler)


# ------------------------------------------------------------ endpoints
@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/jobs")
async def create_job(
    url: str = Form(""),
    file: UploadFile | None = File(None),
    num_clips: int = Form(5),
    duration: int = Form(45),
    voice: str = Form("en-US-GuyNeural"),
    style: str = Form("kinetic"),
    llm_provider: str = Form("auto"),
    whisper_model: str = Form("small"),
    voiceover: str = Form("true"),
    vo_volume: float = Form(1.0),
    bg_volume: float = Form(-1.0),
    content_type: str = Form("auto"),
    trim_silence: str = Form("auto"),
    focus: str = Form(""),
    manual_clips: str = Form(""),
    caption_preset: str = Form("bold"),
    caption_position: str = Form("lower"),
    tts_engine: str = Form("edge"),
    kokoro_voice: str = Form("af_heart"),
):
    # batch mode: one job per URL line; upload = single job
    urls = [u.strip() for u in url.splitlines() if u.strip()]
    inputs: list[str] = []
    if file is not None and file.filename:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        inputs.append(str(dest))
    inputs.extend(urls)
    if not inputs:
        raise HTTPException(400, "provide a URL (one per line for a batch) "
                                 "or upload a file")

    jobs = []
    for input_src in inputs:
        cfg = Config(input=input_src, workdir=str(RUNS_DIR),
                     num_clips=num_clips, duration=duration, voice=voice,
                     style=style, llm_provider=llm_provider,
                     whisper_model=whisper_model,
                     voiceover=voiceover.lower() in ("1", "true", "yes"),
                     vo_volume=vo_volume, bg_audio_volume=bg_volume,
                     content_type=content_type, trim_silence=trim_silence,
                     focus=focus.strip(), manual_clips=manual_clips.strip(),
                     caption_preset=caption_preset,
                     caption_position=caption_position,
                     tts_engine=tts_engine, kokoro_voice=kokoro_voice)
        cfg.min_duration = max(duration - 15, 15)
        cfg.max_duration = duration + 15
        cfg.run_id = derive_run_id(cfg.input)

        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "queued", "run_id": cfg.run_id,
                        "input": input_src, "log": collections.deque(maxlen=400),
                        "manifest": None}
        # JOB_LOCK inside the worker serializes the pipeline runs
        threading.Thread(target=_worker, args=(job_id, cfg), daemon=True).start()
        jobs.append({"job_id": job_id, "run_id": cfg.run_id})

    return {"job_id": jobs[0]["job_id"], "run_id": jobs[0]["run_id"], "jobs": jobs}


def _clip_urls(clips: list[dict], run_id: str) -> list[dict]:
    out = []
    for c in clips:
        c = dict(c)
        if c.get("file"):
            c["url"] = f"/runs/{run_id}/" + c["file"].replace("\\", "/")
        if c.get("thumbs"):
            c["thumbs"] = [f"/runs/{run_id}/clips/{c['clip']}/{t}"
                           for t in c["thumbs"]]
        out.append(c)
    return out


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    resp = {"status": job["status"], "run_id": job["run_id"],
            "stage": job.get("stage"), "stage_detail": job.get("stage_detail"),
            "log": list(job["log"]), "error": job.get("error")}
    if job["manifest"]:
        resp["clips"] = _clip_urls(job["manifest"]["clips"], job["run_id"])
        resp["cost"] = job["manifest"]["saas_cost_equivalent"]
    return JSONResponse(resp)


@app.get("/api/runs")
def list_runs():
    """Library view: every past run that produced a manifest."""
    import json
    runs = []
    for mf in sorted(RUNS_DIR.glob("*/manifest.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        clips = _clip_urls(m.get("clips", []), m.get("run_id", mf.parent.name))
        meta_file = mf.parent / "meta.json"
        try:
            title = json.loads(meta_file.read_text(encoding="utf-8")).get("title", "")
        except Exception:
            title = ""
        runs.append({
            "run_id": m.get("run_id", mf.parent.name),
            "title": title,
            "input": m.get("input", ""),
            "created": m.get("created", ""),
            "settings": m.get("settings", {}),
            "clips": clips,
            "cost": m.get("saas_cost_equivalent", {}).get("total_credits_equivalent"),
        })
    return runs


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a run folder (they hold the downloaded source, so they're big)."""
    import shutil
    if "/" in run_id or "\\" in run_id or run_id.startswith((".", "_")):
        raise HTTPException(400, "bad run id")
    target = RUNS_DIR / run_id
    if not target.is_dir() or not (target / "manifest.json").exists():
        raise HTTPException(404, "unknown run")
    if any(j["run_id"] == run_id and j["status"] in ("queued", "running")
           for j in JOBS.values()):
        raise HTTPException(409, "run is currently in progress")
    shutil.rmtree(target)
    return {"deleted": run_id}


@app.get("/api/youtube/status")
def youtube_status():
    from . import youtube
    return {"configured": youtube.is_configured(),
            "authorized": youtube.is_authorized()}


@app.post("/api/youtube/connect")
def youtube_connect():
    """Runs the one-time OAuth consent flow (opens a browser on this
    machine). Blocks until consent completes."""
    from . import youtube
    try:
        youtube.connect()
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"authorized": True}


@app.post("/api/youtube/upload")
def youtube_upload(run_id: str = Form(...), clip: str = Form(...),
                   privacy: str = Form("private")):
    import json

    from . import youtube
    if privacy not in ("private", "unlisted", "public"):
        raise HTTPException(400, "privacy must be private, unlisted, or public")
    run_dir = RUNS_DIR / run_id
    mf = run_dir / "manifest.json"
    if "/" in run_id or "\\" in run_id or not mf.is_file():
        raise HTTPException(404, "unknown run")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    entry = next((c for c in manifest.get("clips", [])
                  if c.get("clip") == clip and c.get("file")), None)
    if not entry:
        raise HTTPException(404, "unknown clip")
    video = run_dir / entry["file"]
    if not video.is_file():
        raise HTTPException(404, "clip file missing")

    run_title = ""
    meta_file = run_dir / "meta.json"
    if meta_file.is_file():
        run_title = json.loads(meta_file.read_text(encoding="utf-8")).get("title", "")
    title, desc, tags = youtube.build_description(
        entry.get("metadata"), fallback_title=f"{run_title} {clip}".strip())
    try:
        url = youtube.upload(video, title, desc, tags, privacy=privacy)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"url": url, "title": title}


@app.get("/api/voices")
def voices():
    # curated shortlist; full list via `python -m shortsmaker voices`.
    # Multilingual voices first -- newest generation, far more natural.
    return [
        "en-US-AndrewMultilingualNeural", "en-US-BrianMultilingualNeural",
        "en-US-AvaMultilingualNeural", "en-US-EmmaMultilingualNeural",
        "en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-JennyNeural",
        "en-US-AriaNeural", "en-GB-RyanNeural", "en-AU-WilliamNeural",
        "en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "hi-IN-MadhurNeural",
        "hi-IN-SwaraNeural",
    ]


RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    setup_logging()
    log.info("shortsmaker web UI -> http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
