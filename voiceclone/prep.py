"""Préparation automatique d'un enregistrement pour le clonage.

Pipeline : nettoyage (passe-haut, débruitage spectral, volume) -> découpe aux silences ->
notation de chaque morceau -> sélection des meilleurs jusqu'à une durée cible (le meilleur en premier).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

from . import audio

FRAME_MS = 20.0
MIN_SNR_DB = 15.0  # en dessous, la voix est trop noyée dans le bruit pour servir de référence


# ------------------------------------------------------------------ découpe
def smoothed_levels(x: np.ndarray, sr: int, width: int = 5) -> np.ndarray:
    """Niveaux en dB par trame, lissés sur ~100 ms (bords prolongés : un zéro vaudrait 0 dB, très fort)."""
    levels = audio.frame_rms_db(x, sr, FRAME_MS)
    padded = np.pad(levels, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def find_cuts(x: np.ndarray, sr: int, min_s: float, max_s: float) -> list[int]:
    """Positions de coupe (en échantillons) : le passage le plus calme entre min_s et max_s après la coupe précédente."""
    levels = smoothed_levels(x, sr)  # vraies pauses plutôt qu'un creux isolé
    hop = int(sr * FRAME_MS / 1000)
    n_frames = len(levels)
    min_f, max_f = int(min_s * 1000 / FRAME_MS), int(max_s * 1000 / FRAME_MS)
    cuts, start = [], 0
    while n_frames - start > max_f:
        # la coupe doit laisser au moins min_s derrière elle (pas de minuscule dernier morceau)
        window = levels[start + min_f:min(start + max_f, n_frames - min_f)]
        if len(window) == 0:
            break
        cut = start + min_f + int(np.argmin(window))
        cuts.append(cut * hop + hop // 2)
        start = cut
    return cuts


def segment_bounds(x: np.ndarray, sr: int, min_s: float = 4.0, max_s: float = 11.0) -> list[tuple[int, int]]:
    bounds = [0, *find_cuts(x, sr, min_s, max_s), len(x)]
    return list(zip(bounds[:-1], bounds[1:]))


def phrase_bounds(x: np.ndarray, sr: int, floor_db: float, min_s: float = 3.0,
                  max_s: float = 11.0) -> list[tuple[int, int]]:
    """Découpe en phrases naturelles : coupe au milieu de chaque pause >= 250 ms, puis fusionne les
    morceaux trop courts avec leur voisin et redécoupe ceux qui dépassent max_s.

    Plus fin que segment_bounds : une phrase bruitée ou saturée n'est pas noyée dans une bonne.
    """
    hop = int(sr * FRAME_MS / 1000)
    levels = smoothed_levels(x, sr)
    silent = levels < floor_db + 8
    min_run = int(250 / FRAME_MS)
    cuts, i, n = [], 0, len(silent)
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if j - i >= min_run:
                cuts.append(((i + j) // 2) * hop)
            i = j
        else:
            i += 1
    bounds = [0, *[c for c in cuts if 0 < c < len(x)], len(x)]
    pieces = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    # fusion des morceaux trop courts avec le suivant (ou le précédent pour le dernier)
    merged: list[tuple[int, int]] = []
    for a, b in pieces:
        if merged and (merged[-1][1] - merged[-1][0]) / sr < min_s and (b - merged[-1][0]) / sr <= max_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) / sr < min_s \
            and (merged[-1][1] - merged[-2][0]) / sr <= max_s:
        merged[-2:] = [(merged[-2][0], merged[-1][1])]
    # silences retirés aux bords (150 ms gardés), puis redécoupe des morceaux trop longs
    pad = int(0.15 * sr)
    out: list[tuple[int, int]] = []
    for a, b in merged:
        voiced = np.flatnonzero(~silent[a // hop:(b + hop - 1) // hop])
        if len(voiced):
            a, b = max(a, a + voiced[0] * hop - pad), min(b, a + (voiced[-1] + 1) * hop + pad)
        if (b - a) / sr > max_s:
            out += [(a + c, a + d) for c, d in segment_bounds(x[a:b], sr, min_s, max_s)]
        else:
            out.append((a, b))
    return out


def _fade(part: np.ndarray, sr: int) -> np.ndarray:
    fade = int(sr * 0.01)
    if len(part) > 2 * fade:  # petits fondus pour éviter les clics aux coupures
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        part[:fade] *= ramp
        part[-fade:] *= ramp[::-1]
    return part


def split(x: np.ndarray, sr: int, min_s: float = 4.0, max_s: float = 11.0) -> list[np.ndarray]:
    """Découpe un signal en morceaux de min_s à max_s secondes, coupés aux silences, avec fondus."""
    return [_fade(x[a:b].copy(), sr) for a, b in segment_bounds(x, sr, min_s, max_s)]


# --------------------------------------------------------------- nettoyage
def noise_floor_db(x: np.ndarray, sr: int) -> float:
    return float(np.percentile(audio.frame_rms_db(x, sr, FRAME_MS), 10))


def speech_level_db(x: np.ndarray, sr: int) -> float:
    return float(np.percentile(audio.frame_rms_db(x, sr, FRAME_MS), 90))


def denoise(x: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
    """Débruitage spectral (soustraction douce d'un profil de bruit stationnaire).

    Le profil de bruit est estimé par fréquence sur les trames les plus calmes. Le gain est
    lissé et plafonné (-18 dB max) pour éviter le « bruit musical » qui abîmerait le timbre.
    """
    n = 1024 if sr >= 32000 else 512
    _, _, z = stft(x, sr, nperseg=n, noverlap=n * 3 // 4)
    mag = np.abs(z)
    frame_energy = mag.mean(axis=0)
    quiet = frame_energy <= np.percentile(frame_energy, 15)
    noise = (mag[:, quiet].mean(axis=1) if quiet.any() else np.percentile(mag, 10, axis=1))[:, None]
    gain = np.clip(1.0 - strength * 1.5 * noise / np.maximum(mag, 1e-9), 0.125, 1.0)
    gain = uniform_filter(gain, size=(3, 5))  # lissage fréquence x temps
    _, y = istft(z * gain, sr, nperseg=n, noverlap=n * 3 // 4)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def normalize_loudness(x: np.ndarray, target_db: float = -20.0, peak: float = 0.95) -> np.ndarray:
    gain = 10 ** ((target_db - audio.rms_db(x)) / 20)
    y = x * gain
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (peak / m) if m > peak else y).astype(np.float32)


def clean(x: np.ndarray, sr: int, enhance: bool = True, method: str = "auto") -> tuple[np.ndarray, dict]:
    """Nettoie un enregistrement complet. Renvoie le signal et ce qui a été fait.

    method : "auto", "spectral", "deepfilter", "demucs" ou "demucs+deepfilter" (voir voiceclone.enhance).
    """
    from . import enhance as enh

    info = {"noise_floor_db_before": round(noise_floor_db(x, sr), 1), "denoised": False}
    y = audio.highpass(x, sr, 70.0)
    snr = speech_level_db(y, sr) - noise_floor_db(y, sr)
    explicit = method not in ("auto", "spectral")
    # inutile (et risqué pour le timbre) sur un enregistrement déjà propre, sauf méthode choisie exprès
    if enhance and (snr < 45 or explicit):
        y, details = enh.run(y, sr, method, denoise, strength=1.0 if snr < 30 else 0.6)
        info.update(details, denoised=True)
    info["noise_floor_db_after"] = round(noise_floor_db(y, sr), 1)
    return y, info


# ---------------------------------------------------------------- notation
@dataclass
class Segment:
    index: int
    start: float
    end: float
    duration: float
    snr_db: float
    speech_ratio: float
    clipping: float
    score: float
    kept: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def score_segment(seg: np.ndarray, original: np.ndarray, sr: int, floor_db: float) -> dict:
    """Note 0-100 : rapport voix/bruit, part de parole, durée, absence de saturation."""
    duration = len(seg) / sr
    levels = audio.frame_rms_db(seg, sr, FRAME_MS)
    voiced = levels > floor_db + 10
    speech_ratio = float(voiced.mean()) if len(levels) else 0.0
    if voiced.any():
        # bruit « dans » la phrase : les creux entre syllabes, entre le 1er et le dernier son voisé
        idx = np.flatnonzero(voiced)
        local_floor = max(float(np.percentile(levels[idx[0]:idx[-1] + 1], 5)), floor_db)
        snr = float(np.percentile(levels[voiced], 75) - local_floor)
    else:
        snr = 0.0
    clipping = float(np.mean(np.abs(original) > 0.99)) if len(original) else 0.0

    def clip01(v):
        return float(min(1.0, max(0.0, v)))

    s_snr = clip01((snr - 10) / 30)
    s_speech = clip01((speech_ratio - 0.4) / 0.5)
    s_dur = clip01(duration / 4) if duration < 4 else (1.0 if duration <= 11 else clip01(1 - (duration - 11) / 8))
    s_clip = 1 - clip01(clipping * 200)
    score = 100 * (0.45 * s_snr + 0.25 * s_speech + 0.15 * s_dur + 0.15 * s_clip)
    return {"duration": duration, "snr_db": snr, "speech_ratio": speech_ratio, "clipping": clipping, "score": score}


def auto_prepare(x: np.ndarray, sr: int, enhance: bool = True, target_s: float = 30.0,
                 min_s: float = 3.0, max_s: float = 11.0, method: str = "auto",
                 min_snr_db: float = MIN_SNR_DB) -> tuple[list[np.ndarray], dict]:
    """Nettoie, découpe, note et sélectionne. Renvoie les morceaux gardés (meilleur d'abord) et un rapport."""
    x = audio.to_mono(x)
    cleaned, info = clean(x, sr, enhance, method)
    floor = noise_floor_db(cleaned, sr)
    segments: list[Segment] = []
    pieces: dict[int, np.ndarray] = {}
    for i, (a, b) in enumerate(phrase_bounds(cleaned, sr, floor, min_s, max_s)):
        part = cleaned[a:b]
        m = score_segment(part, x[a:b], sr, floor)
        seg = Segment(index=i, start=round(a / sr, 2), end=round(b / sr, 2), duration=round(m["duration"], 2),
                      snr_db=round(m["snr_db"], 1), speech_ratio=round(m["speech_ratio"], 2),
                      clipping=round(m["clipping"], 4), score=round(m["score"], 1))
        if seg.duration < 2:
            seg.reason = "trop court"
        elif seg.speech_ratio < 0.3:
            seg.reason = "presque pas de parole"
        elif seg.clipping > 0.01:
            seg.reason = "son saturé"
        elif seg.snr_db < min_snr_db:
            seg.reason = "trop de bruit"
        segments.append(seg)
        pieces[i] = part

    # Sélection : meilleurs scores d'abord, jusqu'à la durée cible
    total, kept = 0.0, []
    for seg in sorted((s for s in segments if not s.reason), key=lambda s: -s.score):
        if total + seg.duration > target_s and kept:
            seg.reason = "durée cible atteinte (score plus faible)"
            continue
        seg.kept = True
        kept.append(seg)
        total += seg.duration
    if not kept and segments:  # rien d'acceptable : on garde quand même le moins mauvais
        best = max(segments, key=lambda s: s.score)
        best.kept, best.reason = True, f"gardé faute de mieux ({best.reason})"
        kept = [best]

    out = [_fade(normalize_loudness(pieces[s.index].copy()), sr) for s in kept]
    report = {
        "duration": round(len(x) / sr, 2),
        "kept_duration": round(sum(s.duration for s in kept), 2),
        "kept_order": [s.index for s in kept],
        **info,
        "segments": [s.to_dict() for s in segments],
    }
    return out, report
