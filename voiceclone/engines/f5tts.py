"""F5-TTS v1 (flow matching)."""

from __future__ import annotations

import numpy as np

from ..voices import Voice
from .base import Engine, EngineError, to_numpy


class F5TTSEngine(Engine):
    MAX_REF_SECONDS = 12  # F5-TTS est entraîné avec des références courtes

    def load(self) -> None:
        from f5_tts.api import F5TTS

        sub = self.model_dir / "F5TTS_v1_Base"
        ckpt = sub / "model_1250000.safetensors"
        vocab = sub / "vocab.txt"
        vocos = self.model_dir / "vocos"
        self.model = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(ckpt),
            vocab_file=str(vocab) if vocab.exists() else "",
            vocoder_local_path=str(vocos) if (vocos / "config.yaml").exists() else None,
            device=self.device,
        )
        self.sr = int(self.model.target_sample_rate)

    def _reference(self, voice: Voice) -> tuple[str, str]:
        """Référence courte (≤ 12 s) et sa transcription ("" = transcription auto par F5)."""
        from .. import audio

        if not voice.reference_path.exists():
            raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
        x, sr = voice.reference_audio()
        if len(x) <= self.MAX_REF_SECONDS * sr:
            return str(voice.reference_path), voice.transcript
        # Référence trop longue : on prend le premier échantillon complet qui tient en 12 s
        for s in voice.samples:
            if s.duration <= self.MAX_REF_SECONDS:
                return str(voice.dir / s.file), s.transcript
        short = voice.cache_dir / "f5_ref.wav"
        if not short.exists():
            audio.save_wav(short, x[: self.MAX_REF_SECONDS * sr], sr)
        return str(short), ""

    def prepare_voice(self, voice: Voice) -> None:
        self._reference(voice)

    def tts(self, text: str, voice: Voice, language: str = "en", **params) -> tuple[np.ndarray, int]:
        ref_file, ref_text = self._reference(voice)
        wav, sr, _ = self.model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            speed=float(params.get("speed", 1.0)),
            nfe_step=int(params.get("nfe_step", 32)),
            show_info=lambda *a, **k: None,
        )
        return to_numpy(wav), int(sr)
