"""Filigrane audio inaudible et outil de détection (étalement de spectre).

Principe : une séquence pseudo-aléatoire secrète de 4096 échantillons (à 16 kHz), filtrée entre
1 et 4 kHz, est répétée en boucle et ajoutée ~18 dB sous la voix dans cette bande (≈ 35-40 dB sous le signal total) (elle suit
l'enveloppe du signal pour rester masquée). Pour la détection, on replie le signal par blocs de
4096 échantillons (la voix s'annule en moyenne, la séquence s'additionne), puis on cherche la
séquence par corrélation circulaire : un pic net → filigrane présent. Résiste au découpage,
au décalage, au changement de volume et à l'encodage MP3. Il faut quelques secondes d'audio
(fiable à partir de ~5 s de voix) : plus l'extrait est long, plus le score est élevé.

Ce n'est pas une protection cryptographique : il sert à reconnaître ses propres générations.
Si le paquet `perth` (Resemble AI, installé avec Chatterbox) est présent, son détecteur est
aussi consulté.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from . import audio

SR = 16000
BLOCK = 4096
KEY = os.environ.get("VOICECLONE_WATERMARK_KEY", "voiceclone")
STRENGTH_DB = -18.0
THRESHOLD = 6.5  # z-score du pic de corrélation


def _sequence(key: str = KEY) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    pn = np.random.default_rng(seed).choice([-1.0, 1.0], BLOCK)
    # filtrage circulaire dans la bande 1-4 kHz (FFT), puis normalisation
    spec = np.fft.rfft(pn)
    f = np.fft.rfftfreq(BLOCK, 1 / SR)
    spec[(f < 1000) | (f > 4000)] = 0
    seq = np.fft.irfft(spec, BLOCK)
    return (seq / (np.std(seq) + 1e-9)).astype(np.float32)


def _band(x: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, [1000, min(4000, sr / 2 - 100)], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x) if len(x) > 50 else x


def embed(x: np.ndarray, sr: int, key: str = KEY, strength_db: float = STRENGTH_DB) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < sr // 4:
        return x
    seq = _sequence(key)
    n16 = int(np.ceil(len(x) * SR / sr)) + 1
    mark = np.tile(seq, n16 // BLOCK + 1)[:n16]
    mark = audio.resample(mark, SR, sr)[: len(x)]
    if len(mark) < len(x):
        mark = np.pad(mark, (0, len(x) - len(mark)))
    # enveloppe de la voix dans la bande (lissée ~50 ms) : le filigrane reste sous le masquage
    env = np.abs(_band(x, sr))
    sos = butter(1, 20, btype="lowpass", fs=sr, output="sos")
    env = np.maximum(sosfilt(sos, env), 1e-4)
    gain = 10 ** (strength_db / 20)
    return np.clip(x + gain * env * mark, -1.0, 1.0).astype(np.float32)


def score(x: np.ndarray, sr: int, key: str = KEY) -> float:
    x = audio.resample(np.asarray(x, dtype=np.float32), sr, SR)
    if len(x) < BLOCK * 2:
        return 0.0
    x = _band(x, SR)
    # blanchiment grossier : normalisation par l'enveloppe pour égaliser les passages forts / faibles
    sos = butter(1, 20, btype="lowpass", fs=SR, output="sos")
    env = np.maximum(sosfilt(sos, np.abs(x)), 1e-5)
    x = x / env
    blocks = len(x) // BLOCK
    folded = x[: blocks * BLOCK].reshape(blocks, BLOCK).sum(axis=0)
    seq = _sequence(key)
    # corrélation blanchie (GCC-PHAT) limitée à la bande du filigrane : la voix n'est pas un bruit
    # blanc (harmoniques), sans blanchiment elle masquerait le pic
    cross = np.fft.rfft(folded) * np.conj(np.fft.rfft(seq))
    f = np.fft.rfftfreq(BLOCK, 1 / SR)
    cross = np.where((f >= 1000) & (f <= 4000), cross / (np.abs(cross) + 1e-12), 0)
    corr = np.fft.irfft(cross, BLOCK)
    peak = float(np.max(np.abs(corr)))
    rest = np.delete(corr, int(np.argmax(np.abs(corr))))
    return round(peak / (float(np.std(rest)) + 1e-9), 2)


def perth_score(x: np.ndarray, sr: int) -> float | None:
    try:
        import perth  # noqa: F401
    except Exception:
        return None
    try:
        from perth import PerthImplicitWatermarker

        return float(PerthImplicitWatermarker().get_watermark(x, sample_rate=sr))
    except Exception:
        return None


def detect(x: np.ndarray, sr: int) -> dict:
    s = score(x, sr)
    p = perth_score(x, sr)
    return {"score": s, "threshold": THRESHOLD, "detected": s >= THRESHOLD,
            "perth": p, "duration": round(len(x) / sr, 2)}
