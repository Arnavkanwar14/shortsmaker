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
    try:
        with JOB_LOCK:
            job["status"] = "running"
            manifest = run_pipeline(cfg)
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
):
    if not url and (file is None or not file.filename):
        raise HTTPException(400, "provide a URL or upload a file")

    if file is not None and file.filename:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        input_src = str(dest)
    else:
        input_src = url.strip()

    cfg = Config(input=input_src, workdir=str(RUNS_DIR),
                 num_clips=num_clips, duration=duration, voice=voice,
                 style=style, llm_provider=llm_provider,
                 whisper_model=whisper_model)
    cfg.min_duration = max(duration - 15, 15)
    cfg.max_duration = duration + 15
    cfg.run_id = derive_run_id(cfg.input)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "run_id": cfg.run_id,
                    "input": input_src, "log": collections.deque(maxlen=400),
                    "manifest": None}
    threading.Thread(target=_worker, args=(job_id, cfg), daemon=True).start()
    return {"job_id": job_id, "run_id": cfg.run_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    resp = {"status": job["status"], "run_id": job["run_id"],
            "log": list(job["log"]), "error": job.get("error")}
    if job["manifest"]:
        clips = []
        for c in job["manifest"]["clips"]:
            c = dict(c)
            if c.get("file"):
                c["url"] = f"/runs/{job['run_id']}/" + c["file"].replace("\\", "/")
            clips.append(c)
        resp["clips"] = clips
        resp["cost"] = job["manifest"]["saas_cost_equivalent"]
    return JSONResponse(resp)


@app.get("/api/voices")
def voices():
    # curated shortlist; full list via `python -m shortsmaker voices`
    return [
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
