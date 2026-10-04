"""Informations de diagnostic : versions installées, outils système, vérifications courantes."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from importlib import metadata

from . import __version__, config, opus
from . import device as devmod

# (nom affiché, distribution pip, module importable)
PACKAGES = (
    ("PyTorch", "torch", "torch"),
    ("torchaudio", "torchaudio", "torchaudio"),
    ("torchcodec", "torchcodec", "torchcodec"),
    ("transformers", "transformers", "transformers"),
    ("Coqui TTS (XTTS)", "coqui-tts", "TTS"),
    ("Chatterbox", "chatterbox-tts", "chatterbox"),
    ("F5-TTS", "f5-tts", "f5_tts"),
    ("OpenVoice", "MyShell-OpenVoice", "openvoice"),
    ("faster-whisper", "faster-whisper", "faster_whisper"),
    ("Seed-VC", "seed-vc", "seed_vc"),
    ("RVC (rvc-python)", "rvc-python", "rvc_python"),
    ("DeepFilterNet", "deepfilternet", "df"),
    ("Demucs", "demucs", "demucs"),
    ("librosa", "librosa", "librosa"),
    ("PyAV (Opus)", "av", "av"),
    ("sounddevice", "sounddevice", "sounddevice"),
    ("huggingface_hub", "huggingface_hub", "huggingface_hub"),
    ("numpy", "numpy", "numpy"),
    ("scipy", "scipy", "scipy"),
    ("FastAPI", "fastapi", "fastapi"),
    ("setuptools", "setuptools", "setuptools"),
    ("matplotlib", "matplotlib", "matplotlib"),
)


def _version(dist: str, module: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        pass
    if importlib.util.find_spec(module) is None:
        return None
    try:
        return getattr(__import__(module), "__version__", "installé")
    except Exception:
        return "installé (import en échec)"


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=5).stdout
        return out.splitlines()[0].replace("ffmpeg version ", "")[:60] if out else "installé"
    except Exception:
        return "installé"


def _ver(v: str) -> tuple[int, ...]:
    import re

    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def packages() -> list[dict]:
    return [{"name": n, "dist": d, "version": _version(d, m)} for n, d, m in PACKAGES]


def checks(pkgs: list[dict], ffmpeg: str | None) -> list[dict]:
    """Vérifications des problèmes rencontrés le plus souvent."""
    v = {p["dist"]: p["version"] for p in pkgs}
    out = [
        {"ok": bool(ffmpeg), "label": "ffmpeg (formats MP3/M4A/OGG, export MP3)",
         "help": None if ffmpeg else "sudo apt install ffmpeg"},
        {"ok": bool(v.get("torch")), "label": "PyTorch installé",
         "help": None if v.get("torch") else "Voir README : installer torch adapté à votre CUDA"},
        {"ok": opus.available(), "label": "Compression Opus du Live (PyAV)",
         "help": None if opus.available() else "pip install av"},
    ]
    np_v, mpl = v.get("numpy"), v.get("matplotlib")
    if np_v and mpl and _ver(np_v) < (1, 25):
        out.append({"ok": False, "label": f"numpy {np_v} trop ancien pour matplotlib (entraînement XTTS impossible)",
                    "help": "pip install 'numpy==1.26.4' puis python scripts/install.py --isolated rvc"})
    st = v.get("setuptools")
    if v.get("chatterbox-tts") and st and st[0].isdigit() and int(st.split(".")[0]) >= 81:
        out.append({"ok": False, "label": "setuptools < 81 pour Chatterbox (perth / pkg_resources)",
                    "help": "Contourné automatiquement ; sinon : pip install 'setuptools<81'"})
    return out


def report(manager=None) -> dict:
    pkgs = packages()
    ffmpeg = _ffmpeg()
    disk = shutil.disk_usage(config.DATA_DIR) if config.DATA_DIR.exists() else None
    return {
        "voiceclone": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "data_dir": str(config.DATA_DIR),
        "disk_free_gb": round(disk.free / 1024**3, 1) if disk else None,
        "ffmpeg": ffmpeg,
        "system": devmod.system_info(),
        "packages": pkgs,
        "checks": checks(pkgs, ffmpeg),
        "loaded_models": manager.loaded() if manager else [],
    }
