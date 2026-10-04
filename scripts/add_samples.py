#!/usr/bin/env python3
"""Ajoute des fichiers audio présents sur le serveur à une voix, sans passer par le navigateur.

Les fichiers subissent le même traitement qu'un import dans l'interface (nettoyage, référence
recalculée). Rafraîchissez ensuite la page « Mes voix » pour les voir.

Exemples (depuis la racine du dépôt) :
    .venv/bin/python scripts/add_samples.py --list
    .venv/bin/python scripts/add_samples.py "Ma voix" mon_audio_morceaux/*.wav
    .venv/bin/python scripts/add_samples.py "Ma voix" mon_audio.wav --split          # découpe à la volée
    .venv/bin/python scripts/add_samples.py --create "Nouvelle voix" --lang fr --consent mon_audio.wav --split

Si le serveur a été lancé avec VOICECLONE_DATA, définissez la même variable pour ce script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_audio import split  # noqa: E402

from voiceclone import audio, config  # noqa: E402
from voiceclone.voices import VoiceStore  # noqa: E402


def find_voice(store: VoiceStore, key: str):
    for v in store.list():
        if key in (v.id, v.name) or key.lower() == v.name.lower():
            return v
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Ajoute des fichiers audio du serveur à une voix VoiceClone.")
    p.add_argument("voice", nargs="?", help="nom ou identifiant de la voix existante")
    p.add_argument("files", nargs="*", type=Path, help="fichiers audio à ajouter")
    p.add_argument("--list", action="store_true", help="afficher les voix existantes")
    p.add_argument("--create", metavar="NOM", help="créer une nouvelle voix avec ce nom")
    p.add_argument("--lang", default="fr", help="langue de la nouvelle voix (défaut fr)")
    p.add_argument("--consent", action="store_true",
                   help="avec --create : je confirme que c'est ma voix ou que j'ai l'accord de la personne")
    p.add_argument("--split", action="store_true", help="découper chaque fichier aux silences (morceaux ≤ --max s)")
    p.add_argument("--max", type=float, default=11.0, help="avec --split : durée max d'un morceau (défaut 11)")
    p.add_argument("--min", type=float, default=4.0, help="avec --split : durée min d'un morceau (défaut 4)")
    args = p.parse_args()

    config.ensure_dirs()
    store = VoiceStore()
    if args.list or (not args.voice and not args.create):
        voices = store.list()
        print(f"Voix dans {config.VOICES_DIR} :" if voices else f"Aucune voix dans {config.VOICES_DIR}")
        for v in voices:
            print(f"  {v.name!r:30} id={v.id:28} {len(v.samples)} échantillon(s), {v.to_dict()['duration']} s")
        return

    if args.create:
        if not args.consent:
            p.error("--create demande --consent (votre voix, ou accord explicite de la personne)")
        files = ([Path(args.voice)] if args.voice else []) + args.files  # pas de voix existante à nommer
        voice = store.create(args.create, args.lang)
        print(f"Voix créée : {voice.name!r} (id={voice.id})")
    else:
        files = args.files
        voice = find_voice(store, args.voice)
        if voice is None:
            sys.exit(f"Voix introuvable : {args.voice!r}. Voix existantes : "
                     + ", ".join(repr(v.name) for v in store.list()) + " (ou utilisez --create)")
    if not files:
        p.error("aucun fichier à ajouter")

    added = 0
    for f in files:
        try:
            x, sr = audio.load_audio(f)
            pieces = split(x, sr, args.min, args.max) if args.split else [x]
            for i, piece in enumerate(pieces, 1):
                name = f"{f.stem}_{i:02d}.wav" if len(pieces) > 1 else f.name
                voice = store.add_sample(voice.id, audio.to_wav_bytes(piece, sr), source="upload", original_name=name)
                print(f"  + {name}  ({len(piece) / sr:.1f} s)")
                added += 1
        except Exception as exc:  # un fichier illisible ne bloque pas les autres
            print(f"  ! {f} ignoré : {exc}")

    total = voice.to_dict()["duration"]
    print(f"{added} échantillon(s) ajouté(s) à {voice.name!r} — {len(voice.samples)} au total, {total} s.")
    for w in voice.analysis.get("warnings", []):
        print(f"  ⚠ {w}")
    print("Rafraîchissez la page « Mes voix » (F5) pour les voir.")


if __name__ == "__main__":
    main()
