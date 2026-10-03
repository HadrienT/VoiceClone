"""Interface commune des moteurs (TTS, conversion de voix, reconnaissance vocale)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..registry import ModelSpec
from ..voices import Voice


class EngineError(RuntimeError):
    pass


class Engine:
    """Classe de base. Les sous-classes surchargent les capacités qu'elles supportent.

    Conventions : l'audio est en numpy float32 mono ; chaque méthode renvoie (audio, sample_rate).
    """

    def __init__(self, spec: ModelSpec, model_dir: Path, device: str) -> None:
        self.spec = spec
        self.model_dir = Path(model_dir)
        self.device = device

    # -- cycle de vie -----------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        for attr in list(vars(self)):
            if attr not in ("spec", "model_dir", "device"):
                setattr(self, attr, None)

    # -- "entraînement" : pré-calcul du conditionnement d'une voix ----------
    def prepare_voice(self, voice: Voice) -> None:
        """Calcule et met en cache l'empreinte de la voix pour ce modèle (optionnel)."""

    # -- capacités ------------------------------------------------------------
    def tts(self, text: str, voice: Voice, language: str = "fr", **params) -> tuple[np.ndarray, int]:
        raise EngineError(f"{self.spec.name} ne fait pas de synthèse vocale (TTS).")

    def tts_stream(self, text: str, voice: Voice, language: str = "fr", **params) -> Iterator[tuple[np.ndarray, int]]:
        """Synthèse par morceaux (faible latence). Par défaut : une seule pièce."""
        yield self.tts(text, voice, language, **params)

    def convert(self, wav: np.ndarray, sr: int, voice: Voice, **params) -> tuple[np.ndarray, int]:
        raise EngineError(f"{self.spec.name} ne fait pas de conversion de voix (speech-to-speech).")

    def transcribe(self, wav: np.ndarray, sr: int, language: str | None = None) -> str:
        raise EngineError(f"{self.spec.name} ne fait pas de reconnaissance vocale.")


def to_numpy(wav) -> np.ndarray:
    """Convertit un tenseur torch / liste en numpy float32 mono."""
    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def split_text(text: str, max_chars: int = 240) -> list[str]:
    """Découpe un long texte en phrases de taille raisonnable pour les modèles TTS."""
    import re

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?…;:])\s+", text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > max_chars:
            cut = s.rfind(",", 0, max_chars)
            cut = cut if cut > max_chars // 3 else s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]
