"""Utilitaires audio : chargement, rééchantillonnage, nettoyage, analyse, encodage WAV.

Tout l'audio circule en interne en numpy float32 mono dans [-1, 1].
"""

from __future__ import annotations

import io
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioError(ValueError):
    pass


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        # soundfile renvoie (frames, channels)
        x = x.mean(axis=1) if x.shape[1] <= x.shape[0] else x.mean(axis=0)
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def _decode_with_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise AudioError(
            "Format audio non supporté nativement et ffmpeg est introuvable. "
            "Installez ffmpeg ou envoyez un fichier WAV/FLAC/OGG."
        )
    sr = 44100
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=data, capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError(f"Impossible de décoder l'audio : {proc.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr


def load_audio(src: str | Path | bytes, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Charge un fichier (chemin ou octets) en mono float32. Rééchantillonne si target_sr."""
    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        x = to_mono(x)
    except Exception:
        x, sr = _decode_with_ffmpeg(bytes(data))
    if len(x) == 0:
        raise AudioError("Le fichier audio est vide.")
    if target_sr and sr != target_sr:
        x, sr = resample(x, sr, target_sr), target_sr
    return x, int(sr)


def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_wav(path: str | Path, x: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), sr, subtype="PCM_16")
    return path


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))


def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 30.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    frames = len(x) // n
    if frames == 0:
        return np.array([rms_db(x)])
    f = x[: frames * n].reshape(frames, n).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(f * f, axis=1)), 1e-6))


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0, pad_ms: float = 150.0) -> np.ndarray:
    """Coupe le silence au début et à la fin."""
    frame_ms = 20.0
    levels = frame_rms_db(x, sr, frame_ms)
    voiced = np.where(levels > threshold_db)[0]
    if len(voiced) == 0:
        return x
    n = int(sr * frame_ms / 1000)
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * n - pad)
    end = min(len(x), (voiced[-1] + 1) * n + pad)
    return x[start:end]


def compress_pauses(x: np.ndarray, sr: int, threshold_db: float = -45.0, max_pause_ms: float = 600.0) -> np.ndarray:
    """Raccourcit les longues pauses internes (utile pour l'audio de référence)."""
    frame_ms = 20.0
    n = int(sr * frame_ms / 1000)
    levels = frame_rms_db(x, sr, frame_ms)
    max_frames = int(max_pause_ms / frame_ms)
    keep: list[np.ndarray] = []
    silent_run = 0
    for i, lvl in enumerate(levels):
        seg = x[i * n:(i + 1) * n]
        if lvl <= threshold_db:
            silent_run += 1
            if silent_run > max_frames:
                continue
        else:
            silent_run = 0
        keep.append(seg)
    tail = x[len(levels) * n:]
    if len(tail):
        keep.append(tail)
    return np.concatenate(keep) if keep else x


def normalize_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m < 1e-6:
        return x
    return (x * (peak / m)).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff: float = 60.0) -> np.ndarray:
    """Retire le ronflement / DC sous `cutoff` Hz."""
    from scipy.signal import butter, sosfiltfilt

    if len(x) < sr // 10:
        return x
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def analyze(x: np.ndarray, sr: int) -> dict:
    """Statistiques et avertissements de qualité pour un échantillon de voix."""
    duration = len(x) / sr if sr else 0.0
    levels = frame_rms_db(x, sr)
    speech_ratio = float(np.mean(levels > -40.0)) if len(levels) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipping = float(np.mean(np.abs(x) > 0.99)) if len(x) else 0.0
    noise_floor = float(np.percentile(levels, 10)) if len(levels) else -120.0
    warnings: list[str] = []
    if duration < 6:
        warnings.append("Moins de 6 s d'audio : le clonage sera approximatif (10-30 s recommandées).")
    if peak < 0.05:
        warnings.append("Volume très faible : rapprochez-vous du micro.")
    if clipping > 0.001:
        warnings.append("Saturation détectée : baissez le gain du micro.")
    if noise_floor > -45:
        warnings.append("Bruit de fond élevé : enregistrez dans un endroit plus calme.")
    if speech_ratio < 0.4:
        warnings.append("Beaucoup de silence dans l'enregistrement.")
    return {
        "duration": round(duration, 2),
        "sample_rate": sr,
        "peak": round(peak, 3),
        "rms_db": round(rms_db(x), 1),
        "noise_floor_db": round(noise_floor, 1),
        "speech_ratio": round(speech_ratio, 2),
        "warnings": warnings,
    }


def crossfade_concat(chunks: list[np.ndarray], sr: int, fade_ms: float = 20.0) -> np.ndarray:
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    out = chunks[0].astype(np.float32)
    n = int(sr * fade_ms / 1000)
    for c in chunks[1:]:
        f = min(n, len(out), len(c))
        if f > 0:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            mixed = out[-f:] * (1 - ramp) + c[:f] * ramp
            out = np.concatenate([out[:-f], mixed, c[f:]])
        else:
            out = np.concatenate([out, c])
    return out


def encode_mp3(src: str | Path, bitrate: str = "192k") -> bytes:
    """Encode un fichier audio en MP3 avec ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise AudioError("Export MP3 impossible : ffmpeg est introuvable sur le serveur.")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-f", "mp3", "-b:a", bitrate, "pipe:1"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError(f"Échec de l'encodage MP3 : {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout
