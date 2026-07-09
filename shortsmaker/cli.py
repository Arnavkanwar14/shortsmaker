"""shortsmaker CLI.

  shortsmaker run --input <file_or_url> --num-clips 5 --duration 45 \
      --voice en-US-GuyNeural --style kinetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .util import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shortsmaker",
        description="Turn a video (file or URL) into vertical short clips with "
                    "AI voiceover and kinetic captions -- free/open-source stack.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the full pipeline (or selected stages)")
    r.add_argument("--input", required=True, help="local video file or URL")
    r.add_argument("--config", help="JSON config file with defaults to load")
    r.add_argument("--workdir", default="runs", help="parent output folder")
    r.add_argument("--run-id", dest="run_id", help="reuse an existing run folder")
    r.add_argument("--num-clips", dest="num_clips", type=int)
    r.add_argument("--duration", type=int, help="target clip seconds (30-60)")
    r.add_argument("--voice", help="edge-tts voice, e.g. en-US-GuyNeural")
    r.add_argument("--voiceover", type=lambda s: s.lower() in ("1", "true", "yes"),
                   help="true/false: AI voiceover; false keeps original audio "
                        "and captions the source speech")
    r.add_argument("--vo-volume", dest="vo_volume", type=float,
                   help="voiceover loudness, 0..1 (default 1.0)")
    r.add_argument("--bg-volume", dest="bg_audio_volume", type=float,
                   help="original-audio loudness, 0..1 (default: auto -- 0.18 "
                        "under a voiceover, 1.0 without)")
    r.add_argument("--content-type", dest="content_type",
                   choices=["auto", "talk", "action", "funny"],
                   help="tunes highlight scoring; 'action' for gaming/sports")
    r.add_argument("--trim-silence", dest="trim_silence",
                   choices=["auto", "on", "off"],
                   help="jump-cut dead air and um/uh fillers (default auto)")
    r.add_argument("--focus", help="keywords/description of wanted moments, "
                                   "e.g. 'boss fight' or 'funny fails'")
    r.add_argument("--clips", dest="manual_clips",
                   help="explicit spans, skips auto-detection: "
                        "'12:30-13:10, 745-790'")
    r.add_argument("--style", choices=["kinetic", "plain", "none"])
    r.add_argument("--caption-preset", dest="caption_preset",
                   choices=["bold", "beast", "minimal", "karaoke"])
    r.add_argument("--caption-position", dest="caption_position",
                   choices=["lower", "center"])
    r.add_argument("--reframe-style", dest="reframe_style",
                   choices=["tight", "balanced"],
                   help="tight = full-bleed crop (default); balanced = wider, "
                        "less-zoomed crop with blurred top/bottom fill")
    r.add_argument("--kokoro-voice", dest="kokoro_voice",
                   help="local Kokoro voice, e.g. af_heart, am_michael")
    r.add_argument("--clean", type=lambda s: s.lower() in ("1", "true", "yes"),
                   help="true/false: inpaint burned-in captions/watermarks (slow)")
    r.add_argument("--llm-provider", dest="llm_provider",
                   choices=["auto", "ollama", "groq", "gemini", "none"])
    r.add_argument("--ollama-model", dest="ollama_model")
    r.add_argument("--whisper-model", dest="whisper_model")
    r.add_argument("--tts-engine", dest="tts_engine",
                   choices=["edge", "kokoro", "piper"])
    r.add_argument("--piper-model", dest="piper_model")
    r.add_argument("--force", action="store_true",
                   help="re-run stages even if their outputs exist")
    r.add_argument("-v", "--verbose", action="store_true")

    v = sub.add_parser("voices", help="list available edge-tts voices")
    v.add_argument("--lang", default="en", help="language prefix filter")

    w = sub.add_parser("web", help="launch the web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    return p


def cmd_run(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    cfg = Config.load(Path(args.config)) if args.config else Config()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("command", "config", "verbose")}
    cfg.apply_overrides(overrides)
    if cfg.duration:
        cfg.min_duration = max(cfg.duration - 15, 15)
        cfg.max_duration = cfg.duration + 15

    from .pipeline import run
    manifest = run(cfg)
    failed = sum(1 for c in manifest["clips"] if c["status"] != "ok")
    return 1 if failed == len(manifest["clips"]) else 0


def cmd_voices(args: argparse.Namespace) -> int:
    import asyncio
    import edge_tts

    async def _list():
        return await edge_tts.list_voices()

    for v in asyncio.run(_list()):
        if v["Locale"].startswith(args.lang):
            print(f'{v["ShortName"]:34} {v["Gender"]:7} {v["Locale"]}')
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "voices":
        return cmd_voices(args)
    if args.command == "web":
        from .webui import serve
        serve(args.host, args.port)
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
