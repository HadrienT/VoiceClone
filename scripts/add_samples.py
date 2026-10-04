#!/usr/bin/env python3
"""Ajoute des fichiers audio présents sur le serveur à une voix, sans passer par le navigateur.

Les fichiers subissent le même traitement qu'un import dans l'interface (nettoyage, référence
recalculée). Rafraîchissez ensuite la page « Mes voix » pour les voir.

Exemples (depuis la racine du dépôt) :
    .venv/bin/python scripts/add_samples.py --list
    .venv/bin/python scripts/add_samples.py "Ma voix" mon_audio_morceaux/*.wav
    .venv/bin/python scripts/add_samples.py "Ma voix" mon_audio.wav --split          # découpe à la volée
    .venv/bin/python scripts/add_samples.py "Ma voix" mon_audio.wav --auto           # nettoie + garde le meilleur
    .venv/bin/python scripts/add_samples.py --create "Nouvelle voix" --lang fr --consent mon_audio.wav --split
    .venv/bin/python scripts/add_samples.py "Ma voix" longue_interview.wav --training  # audio d'entraînement, sans limite

Si le serveur a été lancé avec VOICECLONE_DATA, définissez la même variable pour ce script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voiceclone import audio, config  # noqa: E402
from voiceclone.prep import auto_prepare, split  # noqa: E402
from voiceclone.voices import VoiceStore, consent_record  # noqa: E402


def find_voice(store: VoiceStore, key: str):
    for v in store.list():
        if key in (v.id, v.name) or key.lower() == v.name.lower():
            return v
    return None


def print_report(f: Path, report: dict) -> None:
    print(f"{f.name} : {report['duration']} s, bruit de fond {report['noise_floor_db_before']} dB"
          + (f" → {report['noise_floor_db_after']} dB (débruité)" if report["denoised"] else ""))
    for s in report["segments"]:
        mark = "✓" if s["kept"] else "✗"
        print(f"  {mark} {s['start']:6.1f}-{s['end']:6.1f} s  score {s['score']:5.1f}  "
              f"voix/bruit {s['snr_db']:4.1f} dB  {'' if s['kept'] else s['reason']}")
    print(f"  → {report['kept_duration']} s gardées")


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
    p.add_argument("--auto", action="store_true",
                   help="préparation automatique : nettoyage, découpe, ne garde que les meilleurs passages")
    p.add_argument("--target", type=float, default=30.0, help="avec --auto : secondes de voix à garder (défaut 30)")
    p.add_argument("--no-denoise", action="store_true", help="avec --auto : ne pas débruiter")
    p.add_argument("--training", action="store_true",
                   help="ajouter à l'audio d'entraînement (affinage XTTS / RVC) : tout est gardé, sans limite de durée")
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
        voice = store.create(args.create, args.lang, consent=consent_record("self", "cli"))
        print(f"Voix créée : {voice.name!r} (id={voice.id})")
    else:
        files = args.files
        voice = find_voice(store, args.voice)
        if voice is None:
            sys.exit(f"Voix introuvable : {args.voice!r}. Voix existantes : "
                     + ", ".join(repr(v.name) for v in store.list()) + " (ou utilisez --create)")
    if not files:
        p.error("aucun fichier à ajouter")

    if args.training:
        for f in files:
            try:
                voice, rep = store.add_training_audio(
                    voice.id, f.read_bytes(), f.name, enhance=not args.no_denoise,
                    progress=lambda x, f=f: print(f"\r  {f.name} : {round(x * 100)} %", end="", flush=True))
                print(f"\r  + {f.name} : {rep['duration']} s analysées → {rep['clips']} extraits, "
                      f"{rep['kept_duration']} s gardées ({rep['rejected']} passages écartés)")
            except Exception as exc:
                print(f"  ! {f} ignoré : {exc}")
        total = sum(t["duration"] for t in voice.training)
        print(f"Audio d'entraînement de {voice.name!r} : {len(voice.training)} extraits, {total / 60:.1f} min.")
        return

    added = 0
    for f in files:
        try:
            x, sr = audio.load_audio(f)
            if args.auto:
                pieces, report = auto_prepare(x, sr, enhance=not args.no_denoise, target_s=args.target)
                print_report(f, report)
            else:
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
