"""Reconnaissance vocale avec faster-whisper."""

from __future__ import annotations

import numpy as np

from .. import audio
from .base import Engine


class WhisperEngine(Engine):
    def load(self) -> None:
        from faster_whisper import WhisperModel

        cuda = self.device.startswith("cuda")
        self.model = WhisperModel(
            str(self.model_dir),
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
        )

    def transcribe(self, wav: np.ndarray, sr: int, language: str | None = None) -> str:
        x = audio.resample(wav, sr, 16000)
        lang = None if not language or language == "auto" else language.split("-")[0]
        segments, _ = self.model.transcribe(x, language=lang, beam_size=1, vad_filter=True,
                                            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
