"""Nettoyage avancé optionnel des enregistrements à l'import.

- DeepFilterNet (`pip install deepfilternet`) : réduction de bruit neuronale spécialisée voix,
  très efficace sur souffle, ventilateur, clavier, rue.
- Demucs (`pip install demucs`) : séparation de sources ; ne garde que la voix d'un extrait
  contenant de la musique ou un fond sonore (vidéo, stream, film).

Ces bibliothèques sont facultatives : si elles manquent ou échouent, on revient au débruitage
spectral intégré (voiceclone.prep.denoise).
"""

from __future__ import annotations

import importlib.util
import logging
import threading

import numpy as np

from . import audio
from . import device as devmod

log = logging.getLogger(__name__)

METHODS = {
    "auto": "Automatique (DeepFilterNet si installé, sinon spectral)",
    "spectral": "Spectral (intégré, rapide)",
    "deepfilter": "DeepFilterNet (bruit de fond)",
    "demucs": "Demucs (retire musique et fond sonore)",
    "demucs+deepfilter": "Demucs puis DeepFilterNet",
}
PIP = {"deepfilter": "pip install deepfilternet", "demucs": "pip install demucs"}

_lock = threading.Lock()
_cache: dict = {}


def available() -> dict[str, bool]:
    df = importlib.util.find_spec("df") is not None and importlib.util.find_spec("torch") is not None
    dm = importlib.util.find_spec("demucs") is not None
    return {"auto": True, "spectral": True, "deepfilter": df, "demucs": dm, "demucs+deepfilter": df and dm}


def describe() -> list[dict]:
    av = available()
    return [{"id": k, "label": v, "available": av[k],
             "pip": None if av[k] else " && ".join(PIP[p] for p in ("demucs", "deepfilter") if p in k and not av[p])}
            for k, v in METHODS.items()]


def deepfilter(x: np.ndarray, sr: int) -> np.ndarray:
    import torch
    from df.enhance import enhance, init_df

    with _lock:
        if "df" not in _cache:
            _cache["df"] = init_df()  # (modèle, état, suffixe) ; télécharge les poids la 1re fois
        model, state, _ = _cache["df"]
        dsr = state.sr()
        y = torch.from_numpy(audio.resample(x, sr, dsr)).unsqueeze(0)
        out = enhance(model, state, y)
    return audio.resample(out.squeeze(0).detach().cpu().numpy().astype(np.float32), dsr, sr)


def demucs_vocals(x: np.ndarray, sr: int) -> np.ndarray:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = devmod.get_device()
    with _lock:
        if "demucs" not in _cache:
            model = get_model("htdemucs")
            model.eval()
            _cache["demucs"] = model
        model = _cache["demucs"]
        msr = model.samplerate
        y = torch.from_numpy(audio.resample(x, sr, msr)).float()
        wav = y.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # (1, 2 canaux, T)
        ref = wav.mean()
        std = wav.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(model, (wav - ref) / std, device=dev if dev != "mps" else "cpu", progress=False)[0]
        vocals = sources[model.sources.index("vocals")] * std + ref
    return audio.resample(vocals.mean(0).cpu().numpy().astype(np.float32), msr, sr)


def run(x: np.ndarray, sr: int, method: str, spectral, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """Applique la méthode demandée ; `spectral(x, sr, strength)` sert de repli. Renvoie (signal, infos)."""
    av = available()
    if method == "auto":
        method = "deepfilter" if av["deepfilter"] else "spectral"
    info: dict = {"method": method}
    y, spectral_done = x, False
    for step in method.split("+"):
        if step != "spectral":
            if not av.get(step):
                info["fallback"] = f"{METHODS[step]} non installé ({PIP[step]}) : débruitage spectral utilisé."
            else:
                try:
                    y = deepfilter(y, sr) if step == "deepfilter" else demucs_vocals(y, sr)
                    continue
                except Exception as exc:  # poids introuvables, mémoire…
                    log.exception("Échec de %s", step)
                    info["fallback"] = f"{METHODS[step]} a échoué ({exc}) : débruitage spectral utilisé."
        if not spectral_done:  # repli (ou choix) : une seule passe spectrale
            y, spectral_done = spectral(y, sr, strength), True
    return np.asarray(y, dtype=np.float32), info
