"""Pipeline orchestrator: runs stages in order, writes the run manifest,
and isolates per-clip failures so one bad segment never kills the batch.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import re
import traceback
from pathlib import Path

from .config import Config
from .stages import assemble, cleanup, highlights, ingest, script_gen, transcribe, tts
from .util import CostLedger, media_duration, read_json, write_json

log = logging.getLogger("shortsmaker")


def source_caption_words(transcript: dict, clip: dict) -> list[dict]:
    """Word timestamps of the original speech inside a clip window,
    shifted so 0 = clip start (what the .ass captioner expects)."""
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if w["start"] >= clip["start"] and w["end"] <= clip["end"]:
                words.append({"start": round(w["start"] - clip["start"], 3),
                              "end": round(w["end"] - clip["start"], 3),
                              "text": w["text"]})
    return words


def derive_run_id(input_str: str) -> str:
    stem = Path(input_str).stem if not input_str.startswith("http") else input_str
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()[:40] or "run"
    digest = hashlib.sha1(input_str.encode()).hexdigest()[:6]
    return f"{slug}-{digest}"


def run(cfg: Config) -> dict:
    if not cfg.run_id:
        cfg.run_id = derive_run_id(cfg.input)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    ledger = CostLedger()
    log.info("=== run %s -> %s ===", cfg.run_id, run_dir)

    # ---- stages 1-3 (whole-video) ----
    video = ingest.run(cfg)
    src_minutes = media_duration(video) / 60
    meta_file = run_dir / "meta.json"
    meta = read_json(meta_file) if meta_file.exists() else {}

    transcript = transcribe.run(cfg, video)
    ledger.add("transcribe", src_minutes)

    if cfg.clean:
        video = cleanup.run(cfg, video)

    clips = highlights.run(cfg, video, transcript)
    ledger.add("highlights", src_minutes)

    # ---- stages 4-6 (per clip, failure-isolated) ----
    manifest_clips = []
    for idx, clip in enumerate(clips, 1):
        clip_dir = run_dir / "clips" / f"clip_{idx:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "clip": f"clip_{idx:02d}",
            "start": clip["start"], "end": clip["end"],
            "duration": round(clip["end"] - clip["start"], 1),
            "score": clip["score"], "reason": clip["reason"],
            "signals": clip.get("signals", {}),
            "status": "ok",
        }
        try:
            if cfg.voiceover:
                context = script_gen.clip_context(transcript, clip, meta)
                script = script_gen.run(cfg, clip, clip_dir, context)
                ledger.add("script", 1)
                entry["script"] = script

                vo_audio, caption_words = tts.run(cfg, script, clip, clip_dir)
                ledger.add("tts", 1)
            else:
                # no voiceover: caption the original speech instead,
                # using the source transcript's word timestamps
                vo_audio = None
                caption_words = source_caption_words(transcript, clip)

            final = assemble.run(cfg, video, clip, clip_dir, vo_audio, caption_words)
            ledger.add("assemble", 1)
            entry["file"] = str(final.relative_to(run_dir))
        except Exception as e:
            log.error("clip %02d FAILED: %s", idx, e)
            log.debug(traceback.format_exc())
            entry["status"] = "failed"
            entry["error"] = str(e)
        manifest_clips.append(entry)

    manifest = {
        "run_id": cfg.run_id,
        "input": cfg.input,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "settings": {"num_clips": cfg.num_clips, "duration": cfg.duration,
                     "voice": cfg.voice, "style": cfg.style,
                     "llm_provider": cfg.llm_provider,
                     "tts_engine": cfg.tts_engine,
                     "voiceover": cfg.voiceover,
                     "content_type": cfg.content_type},
        "clips": manifest_clips,
        "saas_cost_equivalent": ledger.as_dict(),
    }
    write_json(run_dir / "manifest.json", manifest)

    ok = sum(1 for c in manifest_clips if c["status"] == "ok")
    log.info("=== done: %d/%d clips OK | SaaS equivalent ~%s credits saved | %s ===",
             ok, len(manifest_clips), ledger.total, run_dir / "manifest.json")
    return manifest
