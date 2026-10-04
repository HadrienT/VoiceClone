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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voiceclone import audio  # noqa: E402

from voiceclone.prep import split  # noqa: E402


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
