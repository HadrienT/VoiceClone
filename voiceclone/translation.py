"""Traduction vocale : transcription → traduction → synthèse dans la voix clonée."""

from __future__ import annotations

import numpy as np

from .downloads import is_downloaded
from .engines.base import EngineError
from .registry import all_models

base = lambda code: (code or "").split("-")[0].lower()  # noqa: E731


def supports(spec, src: str, tgt: str) -> bool:
    langs = [base(x) for x in spec.languages]
    if spec.engine.endswith("MarianEngine"):
        return langs[:2] == [base(src), base(tgt)]
    return "*" in spec.languages or (base(src) in langs and base(tgt) in langs)


def pick_model(src: str, tgt: str) -> str | None:
    """Premier modèle de traduction prêt (téléchargé + installé) pour cette paire ; les spécialisés d'abord."""
    ready = [s for s in all_models() if "mt" in s.capabilities and is_downloaded(s.id)
             and not s.missing_packages() and supports(s, src, tgt)]
    ready.sort(key=lambda s: 0 if s.engine.endswith("MarianEngine") else 1)
    return ready[0].id if ready else None


def translate_text(manager, text: str, src: str, tgt: str, mt_model_id: str | None = None) -> str:
    if not text or base(src) == base(tgt):
        return text
    mid = mt_model_id or pick_model(src, tgt)
    if not mid:
        raise EngineError(f"Aucun modèle de traduction prêt pour {src} → {tgt}. Téléchargez « NLLB-200 » "
                          "(toutes langues) ou un modèle Opus-MT dans l'onglet Modèles.")
    engine = manager.get(mid, "mt")
    with manager.infer_lock:
        return engine.translate(text, base(src), base(tgt))


def speech_to_translated_text(manager, asr, wav: np.ndarray, sr: int, src: str, tgt: str,
                              mt_model_id: str | None = None) -> tuple[str, str]:
    """Renvoie (texte reconnu, texte traduit). Vers l'anglais sans modèle de traduction : Whisper traduit seul."""
    with manager.infer_lock:
        text = asr.transcribe(wav, sr, src)
    if not text:
        return "", ""
    if base(tgt) == "en" and base(src) != "en" and not mt_model_id and not pick_model(src, tgt) \
            and hasattr(asr, "translate_speech"):
        with manager.infer_lock:
            return text, asr.translate_speech(wav, sr, src)
    return text, translate_text(manager, text, src, tgt, mt_model_id)
