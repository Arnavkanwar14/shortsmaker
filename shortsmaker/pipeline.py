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

from . import edits
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


def parse_manual_clips(spec: str) -> list[tuple[float, float]]:
    """'12:30-13:10, 745-790.5, 1:02:03-1:02:50' -> [(start_s, end_s)]."""
    def ts(s: str) -> float:
        parts = [float(p) for p in s.strip().split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

    spans = []
    for chunk in spec.split(","):
        if "-" not in chunk:
            continue
        a, _, b = chunk.partition("-")
        try:
            s, e = ts(a), ts(b)
        except ValueError:
            continue
        if e > s:
            spans.append((round(s, 2), round(e, 2)))
    return spans


def derive_run_id(input_str: str) -> str:
    stem = Path(input_str).stem if not input_str.startswith("http") else input_str
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()[:40] or "run"
    digest = hashlib.sha1(input_str.encode()).hexdigest()[:6]
    return f"{slug}-{digest}"


def run(cfg: Config, progress=None) -> dict:
    """progress: optional callback(stage, detail) -- stages are
    'ingest' | 'transcribe' | 'highlights' | 'clip' | 'done'."""
    def _p(stage: str, detail: str = "") -> None:
        if progress:
            progress(stage, detail)

    if not cfg.run_id:
        cfg.run_id = derive_run_id(cfg.input)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    ledger = CostLedger()
    log.info("=== run %s -> %s ===", cfg.run_id, run_dir)

    # ---- stages 1-3 (whole-video) ----
    _p("ingest")
    video = ingest.run(cfg)
    src_minutes = media_duration(video) / 60
    meta_file = run_dir / "meta.json"
    meta = read_json(meta_file) if meta_file.exists() else {}

    _p("transcribe")
    transcript = transcribe.run(cfg, video)
    ledger.add("transcribe", src_minutes)

    if cfg.clean:
        video = cleanup.run(cfg, video)
    _p("highlights")

    if cfg.manual_clips:
        spans = parse_manual_clips(cfg.manual_clips)
        if not spans:
            raise ValueError(f"could not parse manual clips: {cfg.manual_clips!r} "
                             "(expected e.g. '12:30-13:10, 745-790')")
        clips = []
        for s, e in spans:
            text, _, _ = highlights._window_texts(transcript["segments"], s, e)
            clips.append({"start": s, "end": e, "score": 1.0, "signals": {},
                          "text": text, "reason": "manual selection"})
        log.info("manual clips: %d spans, auto-detection skipped", len(clips))
    else:
        clips = highlights.run(cfg, video, transcript, meta)
        ledger.add("highlights", src_minutes)

    # ---- stages 4-6 (per clip, failure-isolated) ----
    manifest_clips = []
    for idx, clip in enumerate(clips, 1):
        _p("clip", f"{idx}/{len(clips)}")
        # manual spans get their own span-named dirs so they never collide
        # with (or wrongly reuse) a previous auto run's cached clip outputs
        name = (f"manual_{int(clip['start'])}-{int(clip['end'])}"
                if cfg.manual_clips else f"clip_{idx:02d}")
        clip_dir = run_dir / "clips" / name
        clip_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "clip": name,
            "start": clip["start"], "end": clip["end"],
            "duration": round(clip["end"] - clip["start"], 1),
            "score": clip["score"], "reason": clip["reason"],
            "signals": clip.get("signals", {}),
            "virality": clip.get("virality"),
            "metadata": clip.get("metadata"),
            "status": "ok",
        }
        try:
            # snappy-cut plan: trim dead air + filler words (from the source
            # speech word timestamps); shortens the effective clip length
            words_rel = source_caption_words(transcript, clip)
            keeps = None
            if cfg.trim_silence != "off":
                keeps = edits.plan_cuts(words_rel, clip["end"] - clip["start"],
                                        max_gap=cfg.silence_gap)
            if keeps:
                clip = dict(clip, edited_duration=edits.edited_duration(keeps))
                entry["edited_duration"] = clip["edited_duration"]

            if cfg.voiceover:
                context = script_gen.clip_context(transcript, clip, meta)
                script = script_gen.run(cfg, clip, clip_dir, context)
                ledger.add("script", 1)
                entry["script"] = script

                vo_audio, caption_words = tts.run(cfg, script, clip, clip_dir)
                ledger.add("tts", 1)
            else:
                # no voiceover: caption the original speech instead,
                # remapped onto the compressed timeline if cuts were made
                vo_audio = None
                caption_words = (edits.remap_words(words_rel, keeps)
                                 if keeps else words_rel)

            final = assemble.run(cfg, video, clip, clip_dir, vo_audio,
                                 caption_words, keeps)
            ledger.add("assemble", 1)
            entry["file"] = str(final.relative_to(run_dir))
            entry["thumbs"] = assemble.thumbnails(final, clip_dir)
            if entry["metadata"]:
                m = entry["metadata"]
                (clip_dir / "metadata.txt").write_text(
                    f"{m['title']}\n\n{m['description']}\n\n"
                    + " ".join(f"#{t}" for t in m["hashtags"]),
                    encoding="utf-8")
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
    _p("done")

    ok = sum(1 for c in manifest_clips if c["status"] == "ok")
    log.info("=== done: %d/%d clips OK | SaaS equivalent ~%s credits saved | %s ===",
             ok, len(manifest_clips), ledger.total, run_dir / "manifest.json")
    return manifest
