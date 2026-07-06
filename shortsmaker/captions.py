"""Kinetic .ass caption generator.

Groups TTS word timestamps into short phrases (one on screen at a time),
uses karaoke \\kf tags so the spoken word lights up in the accent color,
and adds a pop-in scale animation per phrase. Keywords get extra pop.
"""
from __future__ import annotations

import re
from pathlib import Path

ACCENT = "&H00FFD400&"      # BGR: vivid cyan highlight for spoken words
PRIMARY = "&H00FFFFFF&"     # white before highlight sweep

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kinetic,Arial Black,88,{accent},{primary},&H00000000&,&H80000000&,-1,0,0,0,100,100,1,0,1,7,3,5,60,60,0,1
Style: Plain,Arial,72,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

KEYWORD = re.compile(
    r"\b(best|worst|most|insane|amazing|incredible|never|always|secret|huge|"
    r"free|wow|crazy|wild|epic|perfect|shocking|unbelievable|\d[\d,.]*%?)\b", re.I)


def _ts(sec: float) -> str:
    sec = max(sec, 0.0)
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def group_phrases(words: list[dict], max_words: int = 4, max_chars: int = 22,
                  max_gap: float = 0.6) -> list[list[dict]]:
    phrases, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x["text"]) + 1 for x in cur) + len(w["text"])
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= max_words or chars > max_chars or gap > max_gap:
                phrases.append(cur)
                cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)
    return phrases


def _phrase_text_kinetic(phrase: list[dict]) -> str:
    """Karaoke sweep + pop-in; keywords uppercased with extra scale punch."""
    parts = [r"{\an5\pos(540,1420)\fad(60,40)"
             r"\t(0,110,\fscx112\fscy112)\t(110,220,\fscx100\fscy100)}"]
    t0 = phrase[0]["start"]
    for w in phrase:
        dur_cs = max(int((w["end"] - w["start"]) * 100), 1)
        text = w["text"]
        if KEYWORD.search(text):
            text = text.upper()
            parts.append(rf"{{\kf{dur_cs}\fscx108\fscy108}}{text} ")
        else:
            parts.append(rf"{{\kf{dur_cs}\fscx100\fscy100}}{text} ")
        # lead-in silence inside the phrase becomes part of the first word
        _ = t0
    return "".join(parts).rstrip()


def build_ass(words: list[dict], style: str = "kinetic") -> str:
    out = [HEADER.format(accent=ACCENT, primary=PRIMARY)]
    if style == "none" or not words:
        return out[0]
    for phrase in group_phrases(words):
        start, end = phrase[0]["start"], phrase[-1]["end"] + 0.08
        if style == "kinetic":
            text = _phrase_text_kinetic(phrase)
            out.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Kinetic,,0,0,0,,{text}\n")
        else:
            plain = " ".join(w["text"] for w in phrase)
            out.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Plain,,0,0,0,,{plain}\n")
    return "".join(out)


def write_ass(words: list[dict], path: Path, style: str = "kinetic") -> Path:
    path.write_text(build_ass(words, style), encoding="utf-8")
    return path
