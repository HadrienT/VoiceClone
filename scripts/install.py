#!/usr/bin/env python3
"""Installe les dépendances des modèles, dans l'environnement courant ou dans un venv isolé.

Exemples :
    python scripts/install.py --list
    python scripts/install.py xtts-v2 whisper-small openvoice-v2
    python scripts/install.py --torch cu124 xtts-v2          # installe d'abord PyTorch pour CUDA 12.4
    python scripts/install.py --isolated chatterbox          # venv dédié data/envs/chatterbox + réglage auto
    python scripts/install.py --all --dry-run                # affiche les commandes sans les lancer

Avec --isolated, le modèle tourne ensuite dans son propre processus Python (voir Diagnostic) :
utile quand deux moteurs veulent des versions incompatibles de torch ou transformers.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voiceclone import config, settings  # noqa: E402
from voiceclone.registry import all_models, get_model  # noqa: E402

BASE_FOR_WORKER = ["numpy", "scipy", "soundfile"]


def installer(python: str) -> list[str]:
    """pip du Python visé ; uv pip si pip est absent (venv créé par uv)."""
    probe = subprocess.run([python, "-m", "pip", "--version"], capture_output=True)
    if probe.returncode == 0:
        return [python, "-m", "pip", "install"]
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", python]
    sys.exit(f"Ni pip ni uv disponibles pour {python}.")


def commands_for(model_id: str, python: str) -> list[list[str]]:
    spec = get_model(model_id)
    out = []
    for part in spec.pip.split("&&"):
        args = shlex.split(part.strip())
        if not args or args[0] != "pip" or args[1:2] != ["install"]:
            continue
        out.append(installer(python) + args[2:])
    return out


def torch_cmd(flavor: str, python: str) -> list[str]:
    index = f"https://download.pytorch.org/whl/{flavor}"
    return installer(python) + ["torch", "torchaudio", "--index-url", index]


def make_venv(model_id: str) -> str:
    env = config.DATA_DIR / "envs" / model_id
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        cmd = (["uv", "venv", "--python", ver, str(env)] if shutil.which("uv")
               else [sys.executable, "-m", "venv", str(env)])
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return str(py)


def run(cmd: list[str], dry: bool) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="identifiants (voir --list)")
    ap.add_argument("--list", action="store_true", help="liste les modèles et leur état")
    ap.add_argument("--all", action="store_true", help="tous les modèles du catalogue")
    ap.add_argument("--torch", metavar="SAVEUR", help="installe PyTorch avant : cu118, cu121, cu124, cu128, cpu…")
    ap.add_argument("--isolated", action="store_true", help="un venv par modèle (data/envs/<id>)")
    ap.add_argument("--dry-run", action="store_true", help="affiche les commandes sans les exécuter")
    a = ap.parse_args()

    if a.list:
        for s in all_models():
            miss = s.missing_packages()
            iso = (settings.get("engine_python") or {}).get(s.id)
            state = "isolé" if iso else ("installé" if not miss else f"manque : {', '.join(miss)}")
            print(f"{s.id:26} {s.name:45} {state}")
        return 0
    ids = [s.id for s in all_models()] if a.all else a.models
    if not ids:
        ap.print_help()
        return 2
    for mid in ids:
        get_model(mid)  # erreur claire si l'identifiant est inconnu

    if not a.isolated:
        if a.torch:
            run(torch_cmd(a.torch, sys.executable), a.dry_run)
        for mid in ids:
            print(f"\n== {mid}")
            for cmd in commands_for(mid, sys.executable):
                run(cmd, a.dry_run)
        print("\nTerminé. Relancez le serveur VoiceClone pour prendre en compte les nouveaux paquets.")
        return 0

    pythons = {}
    for mid in ids:
        print(f"\n== {mid} (environnement isolé)")
        py = make_venv(mid) if not a.dry_run else str(config.DATA_DIR / "envs" / mid / "bin/python")
        if a.torch:
            run(torch_cmd(a.torch, py), a.dry_run)
        run(installer(py) + BASE_FOR_WORKER if not a.dry_run else ["pip", "install", *BASE_FOR_WORKER], a.dry_run)
        for cmd in (commands_for(mid, py) if not a.dry_run else []):
            run(cmd, a.dry_run)
        pythons[mid] = py
    if not a.dry_run:
        settings.update(engine_python={**(settings.get("engine_python") or {}), **pythons})
        print("\nRéglage enregistré : ces modèles tourneront dans leur propre processus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
