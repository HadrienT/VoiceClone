#!/usr/bin/env python3
"""Découpe un enregistrement en morceaux courts, en coupant dans les silences (jamais au milieu d'un mot).

Utile pour F5-TTS, qui utilise un échantillon de 12 s maximum : importez ensuite les morceaux
comme plusieurs échantillons de la même voix (onglet « Mes voix »).

Usage (depuis la racine du dépôt) :
    .venv/bin/python scripts/split_audio.py mon_audio.wav
    .venv/bin/python scripts/split_audio.py mon_audio.mp3 --max 11 --min 4 --out morceaux/

Formats : WAV, FLAC, OGG, MP3… (les autres via ffmpeg s'il est installé).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voiceclone import audio  # noqa: E402

FRAME_MS = 20.0


def find_cuts(x: np.ndarray, sr: int, min_s: float, max_s: float) -> list[int]:
    """Positions de coupe (en échantillons) : le passage le plus calme entre min_s et max_s après la coupe précédente."""
    levels = audio.frame_rms_db(x, sr, FRAME_MS)
    # lissage sur ~100 ms pour viser de vraies pauses plutôt qu'un creux isolé
    levels = np.convolve(levels, np.ones(5) / 5, mode="same")
    hop = int(sr * FRAME_MS / 1000)
    n_frames = len(levels)
    min_f, max_f = int(min_s * 1000 / FRAME_MS), int(max_s * 1000 / FRAME_MS)
    cuts, start = [], 0
    while n_frames - start > max_f:
        # la coupe doit laisser au moins min_s derrière elle (pas de minuscule dernier morceau)
        window = levels[start + min_f:min(start + max_f, n_frames - min_f)]
        if len(window) == 0:
            break
        cut = start + min_f + int(np.argmin(window))
        cuts.append(cut * hop + hop // 2)
        start = cut
    return cuts


def split(x: np.ndarray, sr: int, min_s: float = 4.0, max_s: float = 11.0) -> list[np.ndarray]:
    """Découpe un signal en morceaux de min_s à max_s secondes, coupés aux silences, avec fondus."""
    bounds = [0, *find_cuts(x, sr, min_s, max_s), len(x)]
    fade = int(sr * 0.01)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    parts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        part = x[a:b].copy()
        if len(part) > 2 * fade:  # petits fondus pour éviter les clics aux coupures
            part[:fade] *= ramp
            part[-fade:] *= ramp[::-1]
        parts.append(part)
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description="Découpe un audio en morceaux courts aux silences.")
    p.add_argument("input", type=Path, help="fichier audio à découper")
    p.add_argument("--max", type=float, default=11.0, help="durée max d'un morceau en secondes (défaut 11)")
    p.add_argument("--min", type=float, default=4.0, help="durée min d'un morceau en secondes (défaut 4)")
    p.add_argument("--out", type=Path, help="dossier de sortie (défaut : <nom>_morceaux/ à côté du fichier)")
    args = p.parse_args()
    if not 0 < args.min < args.max:
        p.error("il faut 0 < --min < --max")

    x, sr = audio.load_audio(args.input)
    out_dir = args.out or args.input.with_name(f"{args.input.stem}_morceaux")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = split(x, sr, args.min, args.max)
    print(f"{args.input.name} : {len(x) / sr:.1f} s → {len(parts)} morceau(x) dans {out_dir}/")
    for i, part in enumerate(parts, 1):
        path = audio.save_wav(out_dir / f"{args.input.stem}_{i:02d}.wav", part, sr)
        print(f"  {path.name}  {len(part) / sr:5.1f} s")


if __name__ == "__main__":
    main()
