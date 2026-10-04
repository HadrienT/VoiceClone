"""Chemins et réglages globaux (surchargés par variables d'environnement)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"

DATA_DIR = Path(os.environ.get("VOICECLONE_DATA", ROOT_DIR / "data")).resolve()
MODELS_DIR = DATA_DIR / "models"
VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"

# "auto", "cuda", "cuda:1", "mps" ou "cpu"
DEVICE = os.environ.get("VOICECLONE_DEVICE", "auto")

# Jeton Hugging Face optionnel (dépôts privés / limites de débit)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

# Durée max conservée pour l'audio de référence d'une voix (secondes)
MAX_REFERENCE_SECONDS = float(os.environ.get("VOICECLONE_MAX_REF_SECONDS", "30"))


def ensure_dirs() -> None:
    for d in (MODELS_DIR, VOICES_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def child_env(**extra: str) -> dict:
    """Environnement pour un sous-processus Python (entraînement, moteur isolé).

    Certaines bibliothèques chargées dans le serveur écrivent PYTHONHASHSEED avec une valeur
    invalide (graine négative ou trop grande) : le Python enfant refuse alors de démarrer
    (« PYTHONHASHSEED must be "random" or an integer in range [0; 4294967295] »). On la retire.
    """
    env = {**os.environ, **extra}
    seed = env.get("PYTHONHASHSEED")
    if seed is not None and seed != "random" and not (seed.isdigit() and int(seed) <= 4294967295):
        env.pop("PYTHONHASHSEED")
    return env
