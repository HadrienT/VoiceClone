"""F5-TTS v1 (flow matching)."""

from __future__ import annotations

import importlib.util
import json
import logging

import numpy as np

from .. import audio
from ..voices import Voice
from .base import Engine, EngineError, read_audio_without_torchcodec, to_numpy

log = logging.getLogger(__name__)

# Modèles Whisper du catalogue utilisables pour transcrire la référence, du plus précis au plus léger
WHISPER_IDS = ("whisper-large-v3-turbo", "whisper-small")


class F5TTSEngine(Engine):
    MAX_REF_SECONDS = 12  # F5-TTS est entraîné avec des références courtes

    def load(self) -> None:
        from f5_tts.api import F5TTS
        from f5_tts.infer import utils_infer

        read_audio_without_torchcodec(utils_infer)  # lecture de la référence via soundfile

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
        """Référence courte (≤ 12 s) et sa transcription si on la connaît ("" sinon)."""
        if not voice.reference_path.exists():
            raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
        x, sr = voice.reference_audio()
        if len(x) <= self.MAX_REF_SECONDS * sr:
            return str(voice.reference_path), voice.transcript
        # Référence trop longue : un échantillon complet de 12 s max, de préférence déjà transcrit
        short_samples = [s for s in voice.samples if s.duration <= self.MAX_REF_SECONDS]
        if short_samples:
            best = min(short_samples, key=lambda s: not s.transcript)  # le 1er transcrit, sinon le 1er
            return str(voice.dir / best.file), best.transcript
        short = voice.cache_dir / "f5_ref.wav"
        if not short.exists():
            audio.save_wav(short, x[: self.MAX_REF_SECONDS * sr], sr)
        return str(short), ""

    def _ref_text(self, voice: Voice, ref_file: str, known: str) -> str:
        """Texte prononcé dans la référence, indispensable à F5-TTS.

        On ne laisse jamais F5 transcrire lui-même : son pipeline `transformers` passe par
        torchcodec (souvent incompatible avec le CUDA de torch) et télécharge un Whisper de plus.
        On utilise la transcription de la voix, sinon un Whisper du catalogue, avec cache disque.
        """
        if known.strip():
            return known.strip()
        cache = voice.cache_dir / "f5_ref_text.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("file") == ref_file and data.get("text"):
                return data["text"]
        text = self._transcribe(ref_file, voice.language)
        cache.write_text(json.dumps({"file": ref_file, "text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def _transcribe(self, ref_file: str, language: str) -> str:
        from ..device import cuda_index
        from ..downloads import is_downloaded, model_dir

        available = [m for m in WHISPER_IDS if is_downloaded(m)]
        if not available or importlib.util.find_spec("faster_whisper") is None:
            raise EngineError(
                "F5-TTS a besoin du texte exact prononcé dans l'échantillon. Renseignez-le dans "
                "« Mes voix » → Échantillons & transcription, ou installez faster-whisper et "
                "téléchargez un modèle Whisper (onglet Modèles) pour le transcrire automatiquement."
            )
        from faster_whisper import WhisperModel

        cuda = self.device.startswith("cuda")
        whisper = WhisperModel(str(model_dir(available[0])), device="cuda" if cuda else "cpu",
                               device_index=cuda_index(self.device) if cuda else 0,
                               compute_type="float16" if cuda else "int8")
        try:
            x, _ = audio.load_audio(ref_file, target_sr=16000)
            lang = None if not language or language == "auto" else language.split("-")[0]
            segments, _ = whisper.transcribe(x, language=lang, beam_size=5, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            del whisper
        if not text:
            raise EngineError("Aucune parole détectée dans l'échantillon de référence.")
        log.info("Référence F5 transcrite avec %s : %s", available[0], text)
        return text

    def prepare_voice(self, voice: Voice) -> None:
        ref_file, ref_text = self._reference(voice)
        self._ref_text(voice, ref_file, ref_text)

    def tts(self, text: str, voice: Voice, language: str = "en", **params) -> tuple[np.ndarray, int]:
        ref_file, ref_text = self._reference(voice)
        ref_text = self._ref_text(voice, ref_file, ref_text)
        wav, sr, _ = self.model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            speed=float(params.get("speed", 1.0)),
            nfe_step=int(params.get("nfe_step", 32)),
            show_info=lambda *a, **k: None,
        )
        return to_numpy(wav), int(sr)
