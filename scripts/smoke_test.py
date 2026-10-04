#!/usr/bin/env python3
"""Test de bout en bout sur les VRAIS modèles installés et téléchargés (à lancer sur le serveur GPU).

Pour chaque modèle prêt : chargement, puis chaque capacité (TTS, conversion, transcription) sur
une voix existante. Affiche durées, facteur temps réel (RTF < 1 = plus rapide que le temps réel)
et erreurs ; écrit les sorties dans un dossier pour les écouter.

    python scripts/smoke_test.py                       # tous les modèles prêts, première voix
    python scripts/smoke_test.py --voice ma-voix-1a2b3c --models xtts-v2 openvoice-v2
    python scripts/smoke_test.py --out /tmp/essais --text "Bonjour, ceci est un test."

Code de sortie : 0 si tout passe, 1 si au moins un test échoue.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voiceclone import audio, config  # noqa: E402
from voiceclone.downloads import is_downloaded  # noqa: E402
from voiceclone.manager import EngineManager  # noqa: E402
from voiceclone.registry import all_models  # noqa: E402
from voiceclone.voices import VoiceStore  # noqa: E402

TEXTS = {"fr": "Bonjour ! Ceci est un essai de clonage de voix, pour vérifier que tout fonctionne.",
         "en": "Hello! This is a voice cloning test, to check that everything works.",
         "zh": "你好！这是一个声音克隆测试。"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="identifiants à tester (défaut : tous ceux qui sont prêts)")
    ap.add_argument("--voice", help="identifiant ou nom de la voix (défaut : la plus récente)")
    ap.add_argument("--text", help="texte à synthétiser")
    ap.add_argument("--out", default=str(config.DATA_DIR / "smoke_test"), help="dossier des sorties")
    ap.add_argument("--keep-loaded", action="store_true", help="ne pas décharger entre deux modèles")
    a = ap.parse_args()

    store = VoiceStore()
    voices = store.list()
    if not voices:
        print("Aucune voix : créez-en une dans l'interface ou avec scripts/add_samples.py.")
        return 1
    voice = next((v for v in voices if a.voice in (v.id, v.name)), None) if a.voice else voices[0]
    if voice is None:
        print(f"Voix introuvable : {a.voice}")
        return 1
    ref, ref_sr = voice.reference_audio()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Voix : {voice.name} ({voice.id}, {len(ref) / ref_sr:.1f} s) — sorties dans {out}\n")

    manager = EngineManager()
    specs = [s for s in all_models() if (not a.models or s.id in a.models)]
    results = []

    def record(model, step, ok, seconds=None, audio_s=None, detail=""):
        rtf = f"{seconds / audio_s:.2f}" if seconds and audio_s else ""
        results.append(ok)
        mark = "OK " if ok else "ÉCHEC"
        print(f"  {mark:5} {step:12} {seconds or 0:7.2f} s  RTF {rtf:6} {detail}")

    for spec in specs:
        if not is_downloaded(spec.id) or spec.missing_packages():
            if a.models:
                why = "non téléchargé" if not is_downloaded(spec.id) else f"manque {spec.missing_packages()}"
                print(f"■ {spec.id} : ignoré ({why})")
                results.append(False)
            continue
        print(f"■ {spec.id} — {spec.name}")
        t0 = time.time()
        try:
            engine = manager.get(spec.id)
            record(spec.id, "chargement", True, time.time() - t0, detail=engine.device)
        except Exception as exc:
            record(spec.id, "chargement", False, time.time() - t0, detail=str(exc)[:200])
            continue
        lang = voice.language if voice.language in spec.languages or "*" in spec.languages else \
            (spec.languages[0] if spec.languages else "fr")
        text = a.text or TEXTS.get(lang, TEXTS["en"])
        steps = []
        if "tts" in spec.capabilities:
            steps.append(("tts", lambda: engine.tts(text, voice, lang)))
        if "vc" in spec.capabilities:
            steps.append(("conversion", lambda: engine.convert(ref[: ref_sr * 8], ref_sr, voice)))
        if "asr" in spec.capabilities:
            steps.append(("transcription", lambda: engine.transcribe(ref[: ref_sr * 15], ref_sr, voice.language)))
        for name, fn in steps:
            t0 = time.time()
            try:
                with manager.infer_lock:
                    res = fn()
                dt = time.time() - t0
                if isinstance(res, str):
                    record(spec.id, name, bool(res.strip()), dt, min(15, len(ref) / ref_sr), f"« {res[:80]} »")
                else:
                    wav, sr = res
                    ok = len(wav) > sr * 0.3 and float(abs(wav).max()) > 0.01
                    path = out / f"{spec.id}-{name}.wav"
                    audio.save_wav(path, wav, sr)
                    record(spec.id, name, ok, dt, len(wav) / sr, f"{path.name}" + ("" if ok else " (silence !)"))
            except Exception as exc:
                record(spec.id, name, False, time.time() - t0, detail=f"{type(exc).__name__}: {str(exc)[:200]}")
                traceback.print_exc(limit=3)
        if not a.keep_loaded:
            manager.unload(spec.id)

    if not results:
        print("Aucun modèle prêt (téléchargé + dépendances installées).")
        return 1
    print(f"\n{sum(results)}/{len(results)} vérifications réussies.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
