"""Textes longs : découpage en phrases, pauses explicites, génération phrase par phrase.

Syntaxe des pauses dans le texte : « [pause] » (0,8 s), « [pause 2s] », « [pause 500ms] ».
Une ligne vide entre deux paragraphes ajoute une pause de 0,6 s.
"""

from __future__ import annotations

import re

import numpy as np

from .engines.base import split_text

PAUSE_RE = re.compile(r"\[\s*pause(?:\s+([\d.,]+)\s*(ms|s)?)?\s*\]", re.I)
DEFAULT_PAUSE = 0.8
PARAGRAPH_PAUSE = 0.6
GAP_S = 0.12  # petit silence entre deux phrases


def parse(text: str, max_chars: int = 240) -> list[dict]:
    """Liste de segments {"type": "text", "text"} / {"type": "pause", "seconds"}."""
    items: list[dict] = []

    def pause(sec: float) -> None:
        if items and items[-1]["type"] == "pause":
            items[-1]["seconds"] = round(max(items[-1]["seconds"], sec), 3)
        elif items:
            items.append({"type": "pause", "seconds": round(sec, 3)})

    for p_i, para in enumerate(re.split(r"\n\s*\n", text.strip())):
        if p_i:
            pause(PARAGRAPH_PAUSE)
        pos = 0
        for m in [*PAUSE_RE.finditer(para), None]:
            chunk = para[pos: m.start() if m else len(para)]
            items += [{"type": "text", "text": t} for t in split_text(chunk, max_chars)]
            if m is None:
                break
            sec = DEFAULT_PAUSE
            if m.group(1):
                sec = float(m.group(1).replace(",", "."))
                sec = sec / 1000 if (m.group(2) or "s").lower() == "ms" else sec
            pause(min(sec, 30.0))
            pos = m.end()
    while items and items[-1]["type"] == "pause":
        items.pop()
    return items


def has_markup(text: str) -> bool:
    return bool(PAUSE_RE.search(text))


def assemble(parts: list[tuple[dict, np.ndarray | None]], sr: int) -> np.ndarray:
    """Recolle les phrases générées et les pauses (fondus courts aux jointures)."""
    out: list[np.ndarray] = []
    fade = int(0.008 * sr)
    gap = np.zeros(int(GAP_S * sr), dtype=np.float32)
    prev_text = False
    for seg, wav in parts:
        if seg["type"] == "pause":
            out.append(np.zeros(int(seg["seconds"] * sr), dtype=np.float32))
            prev_text = False
            continue
        if wav is None or not len(wav):
            continue
        w = np.asarray(wav, dtype=np.float32).copy()
        if len(w) > 2 * fade:
            w[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            w[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        if prev_text:
            out.append(gap)
        out.append(w)
        prev_text = True
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
