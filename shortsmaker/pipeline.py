"""Pipeline orchestrator: runs stages in order, writes the run manifest,
and isolates per-clip failures so one bad segment never kills the batch.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import traceback
from pathlib import Path

from . import edits, names, reframe
from .config import Config
from .stages import assemble, cleanup, highlights, ingest, script_gen, transcribe, tts
from .util import CostLedger, media_duration, read_json, write_json

log = logging.getLogger("shortsmaker")

# Every setting that changes what a rendered clip looks/sounds like. The
# per-clip stages (script/tts/assemble) each cache their output purely by
# "does the file exist" -- with no awareness that these settings changed
# since a previous run on the SAME url/run_id, they'd silently keep
# serving a stale render (e.g. an old AI-voiceover clip after switching
# to caption-the-original-voice mode). render_signature() gives each
# clip a fingerprint of its settings so a change forces a fresh render.
RENDER_SETTINGS_KEYS = [
    "voiceover", "tts_engine", "voice", "kokoro_voice", "piper_model",
    "vo_volume", "bg_audio_volume", "style", "caption_preset",
    "caption_position", "reframe_style", "face_crop", "llm_provider",
    "trim_silence", "silence_gap", "trim_bottom_pct",
]

# Bump this whenever script/tts render LOGIC changes in a way that isn't
# captured by any Config field above -- e.g. the beat-aligned narration
# rewrite (v2) needed old cached clips to redo even though no user-facing
# setting changed, or they'd silently keep serving the old drifting-VO render.
RENDER_LOGIC_VERSION = 12


def render_signature(cfg: Config) -> str:
    d = {k: getattr(cfg, k) for k in RENDER_SETTINGS_KEYS}
    d["_v"] = RENDER_LOGIC_VERSION
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


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

    # deterministic proper-name repair (known_names.txt): fixes Whisper's
    # garbled/phonetic misspellings of known names (Blasiken, Gavite ...)
    # in memory each run, BEFORE highlights/script see the text. The
    # cached transcript.json on disk stays untouched (raw ASR output).
    n_fixed = names.correct_transcript(transcript)
    if n_fixed:
        log.info("known-names repair: %d transcript fixes", n_fixed)

    if cfg.clean:
        video = cleanup.run(cfg, video)
    _p("highlights")

    if cfg.whole_clip:
        # re-voice a short you already have: no highlight search, no
        # cropping/selection -- the whole input becomes the one clip.
        # Its transcript (what's actually spoken) still feeds script_gen's
        # prompt below, so the new AI voiceover reacts to the real content
        # instead of writing something generic.
        dur = media_duration(video)
        text, _, _ = highlights._window_texts(transcript["segments"], 0.0, dur)
        clips = [{"start": 0.0, "end": dur, "score": 1.0, "signals": {},
                 "text": text, "reason": "whole clip (no highlight search)"}]
        log.info("whole-clip mode: using the entire %.1fs input, unchanged", dur)
    elif cfg.manual_clips:
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
            "focus_matched": clip.get("focus_matched"),
            "status": "ok",
        }

        # if voiceover/voice/caption/reframe settings changed since this
        # clip was last rendered, its cached script/tts/assemble outputs
        # are stale even though the files still exist -- force a redo for
        # just this clip rather than silently serving the old render.
        # Also fold in the clip's own start/end: clip_01 is an index, not a
        # content identity, so if a highlight-selection change (e.g. --focus)
        # picks different footage for the same index, that must invalidate
        # the cache too even though none of the render SETTINGS changed.
        sig = render_signature(cfg) + f"|{clip['start']:.2f}-{clip['end']:.2f}"
        sig_file = clip_dir / "render_sig.txt"
        stale = not sig_file.is_file() or sig_file.read_text(encoding="utf-8").strip() != sig
        orig_force = cfg.force
        if stale:
            cfg.force = True
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
                # up-front custom script (whole-job) seeds this clip's
                # per-clip file unless one's already there (a UI edit wins)
                if cfg.custom_script.strip():
                    cf = clip_dir / "custom_script.txt"
                    if not cf.exists():
                        cf.write_text(cfg.custom_script.strip(), encoding="utf-8")
                context = script_gen.clip_context(transcript, clip, meta)
                clip_len = clip.get("edited_duration") or (clip["end"] - clip["start"])
                beats = (edits.plan_beats(transcript["segments"], clip["start"],
                                          clip["end"], keeps, span=script_gen.BEAT_SPAN)
                         if clip_len >= script_gen.BEAT_THRESHOLD else None)
                script = script_gen.run(cfg, clip, clip_dir, context, beats)
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

            # auto-generate the subject-following reframe track once per
            # clip (unless the user already hand-edited one, which we never
            # overwrite). assemble reads reframe.json from clip_dir.
            # only generate when missing: never clobber a hand-edited track,
            # and re-detection is deterministic so there's nothing to gain by
            # regenerating. To force fresh detection, delete reframe.json.
            if cfg.face_crop:
                reframe_file = clip_dir / "reframe.json"
                if not reframe_file.exists():
                    subs = reframe.detect_subjects(video, clip["start"], clip["end"])
                    track = reframe.auto_track(subs, clip["end"] - clip["start"])
                    write_json(reframe_file, track)

            final = assemble.run(cfg, video, clip, clip_dir, vo_audio,
                                 caption_words, keeps)
            ledger.add("assemble", 1)
            entry["file"] = str(final.relative_to(run_dir))
            entry["thumbs"] = assemble.thumbnails(final, clip_dir)
            if entry["metadata"]:
                m = entry["metadata"]
                # description already includes the hashtag block; add the
                # separate YouTube keyword-tags line for copy/paste
                tags_line = ", ".join(m.get("tags", []))
                (clip_dir / "metadata.txt").write_text(
                    f"TITLE:\n{m['title']}\n\nDESCRIPTION:\n{m['description']}\n\n"
                    f"TAGS:\n{tags_line}\n",
                    encoding="utf-8")
            sig_file.write_text(sig, encoding="utf-8")
        except Exception as e:
            log.error("clip %02d FAILED: %s", idx, e)
            log.debug(traceback.format_exc())
            entry["status"] = "failed"
            entry["error"] = str(e)
        finally:
            cfg.force = orig_force
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


def rerender_clip(run_dir: Path, clip_name: str, redo_script: bool = False) -> dict:
    """Re-render ONE clip from a finished run's cached artifacts -- used by the
    web editor after a reframe-track or custom-script edit, so a tweak costs
    one clip's ffmpeg pass, not a whole pipeline re-run.

    Reads the run's saved config + transcript; recomputes keeps the same way
    run() does; reuses the clip's on-disk reframe.json / custom_script.txt
    (whatever the editor just wrote). redo_script=True also clears the cached
    script/beats/voiceover so the narration is regenerated (a script edit);
    a pure reframe edit leaves the voiceover untouched and only re-runs
    assemble. Returns the updated manifest clip entry.
    """
    cfg = Config.load(run_dir / "config.json")
    cfg.force = True
    transcript = read_json(run_dir / "transcript.json")
    meta = read_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}
    manifest = read_json(run_dir / "manifest.json")
    entry = next((c for c in manifest["clips"] if c.get("clip") == clip_name), None)
    if entry is None:
        raise ValueError(f"clip {clip_name} not in manifest")

    video = (run_dir / "normalized.mp4")
    if not video.is_file():
        video = ingest.run(cfg)          # falls back to re-resolving the source
    clip_dir = run_dir / "clips" / clip_name

    text, _, _ = highlights._window_texts(transcript["segments"],
                                          entry["start"], entry["end"])
    clip = {"start": entry["start"], "end": entry["end"], "score": entry.get("score", 1.0),
            "signals": entry.get("signals", {}), "text": text,
            "reason": entry.get("reason", "")}

    words_rel = source_caption_words(transcript, clip)
    keeps = None
    if cfg.trim_silence != "off":
        keeps = edits.plan_cuts(words_rel, clip["end"] - clip["start"],
                                max_gap=cfg.silence_gap)
    if keeps:
        clip = dict(clip, edited_duration=edits.edited_duration(keeps))

    if redo_script:
        for f in ("script.txt", "beats.json", "voiceover.mp3", "vo_words.json"):
            (clip_dir / f).unlink(missing_ok=True)
    (clip_dir / "final.mp4").unlink(missing_ok=True)

    if cfg.voiceover:
        context = script_gen.clip_context(transcript, clip, meta)
        clip_len = clip.get("edited_duration") or (clip["end"] - clip["start"])
        beats = (edits.plan_beats(transcript["segments"], clip["start"], clip["end"],
                                  keeps, span=script_gen.BEAT_SPAN)
                 if clip_len >= script_gen.BEAT_THRESHOLD else None)
        script = script_gen.run(cfg, clip, clip_dir, context, beats)
        entry["script"] = script
        vo_audio, caption_words = tts.run(cfg, script, clip, clip_dir)
    else:
        vo_audio = None
        caption_words = (edits.remap_words(words_rel, keeps) if keeps else words_rel)

    final = assemble.run(cfg, video, clip, clip_dir, vo_audio, caption_words, keeps)
    entry["file"] = str(final.relative_to(run_dir))
    entry["thumbs"] = assemble.thumbnails(final, clip_dir)
    entry["status"] = "ok"
    write_json(run_dir / "manifest.json", manifest)
    return entry
