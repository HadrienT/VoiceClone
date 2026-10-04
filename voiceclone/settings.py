"""Réglages persistants modifiables depuis l'interface (data/settings.json).

Les variables d'environnement servent de valeurs par défaut ; ce qui est enregistré depuis
l'interface a priorité.
"""

from __future__ import annotations

import json
import os
import threading

from . import config

DEFAULTS = {
    # 0 = illimité. Au-delà, le modèle utilisé il y a le plus longtemps est déchargé.
    "max_loaded_models": int(os.environ.get("VOICECLONE_MAX_LOADED", "0")),
    # Décharge automatiquement les modèles inutilisés si la mémoire GPU libre ne suffit pas au suivant
    "auto_unload": os.environ.get("VOICECLONE_AUTO_UNLOAD", "1") != "0",
    # Filigrane inaudible ajouté à tout ce qui est généré (texte → voix, voix → voix, livres audio)
    "watermark": os.environ.get("VOICECLONE_WATERMARK", "0") == "1",
    # Python à utiliser par modèle (isolation) : {"model_id": "/chemin/vers/venv/bin/python"}
    "engine_python": {},
    # Entraînement RVC : dossier d'Applio et son Python (vide = détection automatique)
    "applio_dir": os.environ.get("VOICECLONE_APPLIO_DIR", ""),
    "applio_python": os.environ.get("VOICECLONE_APPLIO_PYTHON", ""),
}

_lock = threading.Lock()


def _path():
    return config.DATA_DIR / "settings.json"


def load() -> dict:
    try:
        saved = json.loads(_path().read_text(encoding="utf-8"))
    except Exception:
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def get(key: str):
    return load()[key]


def update(**fields) -> dict:
    with _lock:
        cur = load()
        for k, v in fields.items():
            if k in DEFAULTS and v is not None:
                cur[k] = type(DEFAULTS[k])(v) if not isinstance(DEFAULTS[k], dict) else dict(v)
        _path().parent.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        return cur
