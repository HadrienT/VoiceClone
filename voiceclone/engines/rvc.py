"""RVC (Retrieval-based Voice Conversion) : conversion avec un modèle entraîné sur UNE voix.

Contrairement aux modèles zero-shot, RVC demande un modèle par voix (.pth + .index facultatif),
entraîné avec Applio / RVC WebUI (ou depuis l'onglet Entraînement). Importez-le dans la voix ;
le moteur l'utilise alors pour la conversion (Voix → Voix et Live). Très bonne qualité et rapide.

Bibliothèque : `rvc-python` (épingle numpy<=1.23.5 et fairseq) → à installer de préférence dans
un environnement isolé : python scripts/install.py --isolated rvc
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .. import audio
from ..voices import Voice
from .base import Engine, EngineError, ensure_pkg_resources


def voice_model(voice: Voice) -> tuple[Path, Path | None, str]:
    """(.pth, .index ou None, version) du modèle RVC attaché à la voix."""
    rvc = (voice.settings or {}).get("rvc") or {}
    if not rvc.get("pth") or not (voice.dir / rvc["pth"]).exists():
        raise EngineError(f"La voix « {voice.name} » n'a pas de modèle RVC : importez un fichier .pth "
                          "(onglet Entraînement) ou entraînez-en un avec Applio.")
    index = voice.dir / rvc["index"] if rvc.get("index") and (voice.dir / rvc["index"]).exists() else None
    return voice.dir / rvc["pth"], index, rvc.get("version", "v2")


class RVCEngine(Engine):
    def load(self) -> None:
        ensure_pkg_resources()  # pyworld (dépendance de RVC) importe encore pkg_resources
        import rvc_python
        from rvc_python.infer import RVCInference

        # Modèles de base (HuBERT, RMVPE) téléchargés par l'onglet Modèles : on les place là où
        # rvc-python les cherche pour éviter un second téléchargement
        base = Path(rvc_python.__file__).parent / "base_model"
        base.mkdir(exist_ok=True)
        for name in ("hubert_base.pt", "rmvpe.pt"):
            src = self.model_dir / name
            if src.exists() and not (base / name).exists():
                try:
                    os.symlink(src, base / name)
                except OSError:
                    shutil.copy2(src, base / name)
        dev = self.device if ":" in self.device or self.device == "cpu" else f"{self.device}:0"
        self.rvc = RVCInference(models_dir=str(self.model_dir / "voices"), device="cpu:0" if dev == "cpu" else dev)
        self._loaded: str | None = None

    def _use(self, voice: Voice) -> None:
        pth, index, version = voice_model(voice)
        key = f"{pth}:{pth.stat().st_mtime}"
        if self._loaded != key:
            self.rvc.load_model(str(pth), version=version, index_path=str(index) if index else "")
            self._loaded = key

    def prepare_voice(self, voice: Voice) -> None:
        self._use(voice)

    def convert(self, wav: np.ndarray, sr: int, voice: Voice, **params) -> tuple[np.ndarray, int]:
        self._use(voice)
        _, index, _ = voice_model(voice)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            audio.save_wav(path, audio.resample(wav, sr, 16000), 16000)
            out = self.rvc.vc.vc_single(
                sid=0, input_audio_path=path, f0_up_key=int(params.get("pitch", 0)),
                f0_method=str(params.get("f0_method", "rmvpe")), file_index=str(index) if index else "",
                index_rate=float(params.get("index_rate", 0.5)) if index else 0.0,
                filter_radius=3, resample_sr=0, rms_mix_rate=float(params.get("rms_mix_rate", 0.25)),
                protect=float(params.get("protect", 0.33)), f0_file="", file_index2="")
        finally:
            os.unlink(path)
        if isinstance(out, tuple):  # rvc-python renvoie (message, (None, None)) en cas d'erreur
            raise EngineError(f"RVC : {out[0]}")
        y = np.asarray(out)
        y = y.astype(np.float32) / 32768.0 if y.dtype == np.int16 else y.astype(np.float32)
        return y, int(self.rvc.vc.tgt_sr)
