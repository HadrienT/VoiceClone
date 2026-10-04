"""Reconnaissance vocale avec faster-whisper."""

from __future__ import annotations

import numpy as np

from .. import audio
from ..device import cuda_index
from .base import Engine


class WhisperEngine(Engine):
    def load(self) -> None:
        from faster_whisper import WhisperModel

        cuda = self.device.startswith("cuda")
        self.model = WhisperModel(
            str(self.model_dir),
            device="cuda" if cuda else "cpu",
            device_index=cuda_index(self.device) if cuda else 0,
            compute_type="float16" if cuda else "int8",
        )

    def transcribe(self, wav: np.ndarray, sr: int, language: str | None = None) -> str:
        x = audio.resample(wav, sr, 16000)
        lang = None if not language or language == "auto" else language.split("-")[0]
        text = self._run(x, lang, vad=True)
        if not text:  # le filtre de silence peut tout écarter sur un extrait court ou peu fort
            text = self._run(x, lang, vad=False)
        return text

    def _run(self, x: np.ndarray, lang: str | None, vad: bool) -> str:
        segments, _ = self.model.transcribe(x, language=lang, beam_size=1, vad_filter=vad,
                                            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
