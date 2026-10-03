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
