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

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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
    reframe_style: str = Form("tight"),
    whole_clip: str = Form("false"),
    trim_bottom_pct: float = Form(0.0),
    custom_script: str = Form(""),
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
                     tts_engine=tts_engine, kokoro_voice=kokoro_voice,
                     reframe_style=reframe_style,
                     whole_clip=whole_clip.lower() in ("1", "true", "yes"),
                     trim_bottom_pct=trim_bottom_pct,
                     custom_script=custom_script.strip())
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


# ------------------------------------------------------- per-clip editor
def _locked_rerender(run_dir: Path, clip: str, redo_script: bool):
    """Re-render one clip, but never block: the pipeline runs one job at a
    time, so if a full run is in progress, fail fast with a clear message
    instead of hanging until the connection times out (a real bug hit when
    editing during a run)."""
    from .pipeline import rerender_clip
    if not JOB_LOCK.acquire(timeout=1.0):
        raise HTTPException(409, "a run is in progress -- wait for it to "
                                 "finish, then save your edit")
    try:
        entry = rerender_clip(run_dir, clip, redo_script=redo_script)
    finally:
        JOB_LOCK.release()
    return _clip_urls([entry], run_dir.name)[0]


def _run_dir(run_id: str) -> Path:
    if "/" in run_id or "\\" in run_id or run_id.startswith((".", "_")):
        raise HTTPException(400, "bad run id")
    d = RUNS_DIR / run_id
    if not (d / "manifest.json").is_file():
        raise HTTPException(404, "unknown run")
    return d


@app.get("/api/run/{run_id}/clip/{clip}/edit")
def clip_edit_data(run_id: str, clip: str):
    """Everything the in-browser editor needs: source video URL + dimensions,
    the clip's time window, the current reframe track and script."""
    import json
    from .util import ffprobe_video
    run_dir = _run_dir(run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = next((c for c in manifest["clips"] if c.get("clip") == clip), None)
    if entry is None:
        raise HTTPException(404, "unknown clip")
    clip_dir = run_dir / "clips" / clip
    src = run_dir / "normalized.mp4"
    info = ffprobe_video(str(src)) if src.is_file() else {"width": 0, "height": 0}
    track_f = clip_dir / "reframe.json"
    script_f = clip_dir / "custom_script.txt"
    if not script_f.exists():
        script_f = clip_dir / "script.txt"
    return {
        "source_url": f"/runs/{run_id}/normalized.mp4",
        "src_w": info["width"], "src_h": info["height"],
        "start": entry["start"], "end": entry["end"],
        "out_w": 1080, "out_h": 1920,
        "reframe": json.loads(track_f.read_text(encoding="utf-8")) if track_f.exists() else [],
        "script": script_f.read_text(encoding="utf-8") if script_f.exists() else "",
    }


@app.post("/api/run/{run_id}/clip/{clip}/reframe")
def save_reframe(run_id: str, clip: str, body: dict = Body(...)):
    """Save a hand-edited reframe track and re-render just this clip (no
    voiceover regeneration -- framing only)."""
    import json
    run_dir = _run_dir(run_id)
    track = body.get("track")
    if not isinstance(track, list) or not track:
        raise HTTPException(400, "track must be a non-empty list")
    for k in track:
        if not all(x in k for x in ("t", "zoom", "cx", "cy")):
            raise HTTPException(400, "each keyframe needs t, zoom, cx, cy")
    clip_dir = run_dir / "clips" / clip
    if not clip_dir.is_dir():
        raise HTTPException(404, "unknown clip")
    (clip_dir / "reframe.json").write_text(json.dumps(track), encoding="utf-8")
    return _locked_rerender(run_dir, clip, redo_script=False)


@app.post("/api/run/{run_id}/clip/{clip}/script")
def save_script(run_id: str, clip: str, body: dict = Body(...)):
    """Save a custom narration script and re-voice + re-render this clip."""
    run_dir = _run_dir(run_id)
    script = (body.get("script") or "").strip()
    clip_dir = run_dir / "clips" / clip
    if not clip_dir.is_dir():
        raise HTTPException(404, "unknown clip")
    if script:
        (clip_dir / "custom_script.txt").write_text(script, encoding="utf-8")
    else:
        # empty = revert to the AI-generated script
        (clip_dir / "custom_script.txt").unlink(missing_ok=True)
    return _locked_rerender(run_dir, clip, redo_script=True)


RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    setup_logging()
    log.info("shortsmaker web UI -> http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
