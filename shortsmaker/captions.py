"""Kinetic .ass caption generator.

Groups TTS word timestamps into short phrases (one on screen at a time),
uses karaoke \\kf tags so the spoken word lights up in the accent color,
and adds a pop-in scale animation per phrase. Keywords get extra pop.

Visuals come from a preset (font, size, colors, keyword treatment) plus a
position (lower third or center). Colors are ASS &HAABBGGRR& -- BGR order.
"""
from __future__ import annotations

import re
from pathlib import Path

# accent = color the word sweeps TO while spoken; base = color before that
PRESETS = {
    "bold": {          # the original look: heavy white with a cyan sweep
        "font": "Arial Black", "size": 88, "accent": "&H00FFD400&",
        "base": "&H00FFFFFF&", "outline": 7, "upper_keywords": True,
    },
    "beast": {         # oversized yellow-sweep, everything shouts
        "font": "Arial Black", "size": 100, "accent": "&H0000D4FF&",
        "base": "&H00FFFFFF&", "outline": 9, "upper_keywords": True,
    },
    "minimal": {       # lighter face, white sweep, no uppercasing
        "font": "Arial", "size": 68, "accent": "&H00FFFFFF&",
        "base": "&H00B8B8B8&", "outline": 4, "upper_keywords": False,
    },
    "karaoke": {       # classic green sweep
        "font": "Arial Black", "size": 84, "accent": "&H0084DC3D&",
        "base": "&H00FFFFFF&", "outline": 6, "upper_keywords": False,
    },
}

POSITIONS = {"lower": 1420, "center": 980}

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kinetic,{font},{size},{accent},{base},&H00000000&,&H80000000&,-1,0,0,0,100,100,1,0,1,{outline},3,5,60,60,0,1
Style: Plain,{font},{plain_size},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,120,1

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


def _phrase_text_kinetic(phrase: list[dict], y: int, upper_keywords: bool) -> str:
    """Karaoke sweep + pop-in; keywords optionally uppercased with extra punch."""
    parts = [rf"{{\an5\pos(540,{y})\fad(60,40)"
             r"\t(0,110,\fscx112\fscy112)\t(110,220,\fscx100\fscy100)}"]
    for w in phrase:
        dur_cs = max(int((w["end"] - w["start"]) * 100), 1)
        text = w["text"]
        if KEYWORD.search(text):
            if upper_keywords:
                text = text.upper()
            parts.append(rf"{{\kf{dur_cs}\fscx108\fscy108}}{text} ")
        else:
            parts.append(rf"{{\kf{dur_cs}\fscx100\fscy100}}{text} ")
    return "".join(parts).rstrip()


def build_ass(words: list[dict], style: str = "kinetic",
              preset: str = "bold", position: str = "lower") -> str:
    p = PRESETS.get(preset, PRESETS["bold"])
    y = POSITIONS.get(position, POSITIONS["lower"])
    out = [HEADER.format(font=p["font"], size=p["size"], accent=p["accent"],
                         base=p["base"], outline=p["outline"],
                         plain_size=max(p["size"] - 16, 56))]
    if style == "none" or not words:
        return out[0]
    for phrase in group_phrases(words):
        start, end = phrase[0]["start"], phrase[-1]["end"] + 0.08
        if style == "kinetic":
            text = _phrase_text_kinetic(phrase, y, p["upper_keywords"])
            out.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Kinetic,,0,0,0,,{text}\n")
        else:
            plain = " ".join(w["text"] for w in phrase)
            out.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Plain,,0,0,0,,{plain}\n")
    return "".join(out)


def write_ass(words: list[dict], path: Path, style: str = "kinetic",
              preset: str = "bold", position: str = "lower") -> Path:
    path.write_text(build_ass(words, style, preset, position), encoding="utf-8")
    return path
