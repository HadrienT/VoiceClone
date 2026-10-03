"""Chatterbox (Resemble AI) : TTS anglais + conversion de voix, et TTS multilingue."""

from __future__ import annotations

import numpy as np

from .. import audio
from ..voices import Voice
from .base import Engine, EngineError, split_text, to_numpy


class _ChatterboxTTSBase(Engine):
    """Logique commune : cache des conditionnements par voix + découpage du texte."""

    model_cls_path: tuple[str, str] = ("", "")

    def _load_tts(self):
        import importlib

        module, cls = self.model_cls_path
        model_cls = getattr(importlib.import_module(module), cls)
        self.model = model_cls.from_local(self.model_dir, self.device)
        self.sr = int(self.model.sr)
        self._active_voice: str | None = None

    def _set_voice(self, voice: Voice, exaggeration: float) -> None:
        """Charge (ou calcule puis met en cache) les conditionnements de la voix."""
        key = f"{voice.id}:{voice.updated_at}"
        if self._active_voice == key:
            return
        if not voice.reference_path.exists():
            raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
        conds_cls = type(self.model).__module__
        import importlib

        Conditionals = importlib.import_module(conds_cls).Conditionals
        cache = voice.cache_dir / f"{self.spec.id}.conds.pt"
        if cache.exists():
            self.model.conds = Conditionals.load(cache, map_location=self.device).to(self.device)
        else:
            self.model.prepare_conditionals(str(voice.reference_path), exaggeration=exaggeration)
            self.model.conds.save(cache)
        self._active_voice = key

    def prepare_voice(self, voice: Voice) -> None:
        self._active_voice = None
        self._set_voice(voice, 0.5)

    def _generate(self, text: str, language: str, params: dict):
        raise NotImplementedError

    def tts(self, text: str, voice: Voice, language: str = "fr", **params) -> tuple[np.ndarray, int]:
        self._set_voice(voice, float(params.get("exaggeration", 0.5)))
        pieces = []
        for chunk in split_text(text, 280):
            pieces.append(to_numpy(self._generate(chunk, language, params)))
        if not pieces:
            raise EngineError("Texte vide.")
        return audio.crossfade_concat(pieces, self.sr, 10), self.sr

    def tts_stream(self, text, voice, language="fr", **params):
        self._set_voice(voice, float(params.get("exaggeration", 0.5)))
        for chunk in split_text(text, 200):
            yield to_numpy(self._generate(chunk, language, params)), self.sr

    @staticmethod
    def _common(params: dict) -> dict:
        return {
            "exaggeration": float(params.get("exaggeration", 0.5)),
            "cfg_weight": float(params.get("cfg_weight", 0.5)),
            "temperature": float(params.get("temperature", 0.8)),
        }


class ChatterboxMultilingualEngine(_ChatterboxTTSBase):
    model_cls_path = ("chatterbox.mtl_tts", "ChatterboxMultilingualTTS")

    def load(self) -> None:
        self._load_tts()

    def _generate(self, text, language, params):
        lang = (language or "fr").split("-")[0].lower()
        return self.model.generate(text, language_id=lang, **self._common(params))


class ChatterboxEngine(_ChatterboxTTSBase):
    """Chatterbox anglais (TTS) + ChatterboxVC (speech-to-speech)."""

    model_cls_path = ("chatterbox.tts", "ChatterboxTTS")

    def load(self) -> None:
        # Le TTS est chargé à la demande pour économiser la VRAM en mode live (VC seul)
        self.model = None
        self.vc = None
        self._active_voice = None
        self._vc_voice: str | None = None

    def _ensure_tts(self) -> None:
        if self.model is None:
            self._load_tts()

    def _ensure_vc(self) -> None:
        if self.vc is None:
            from chatterbox.vc import ChatterboxVC

            self.vc = ChatterboxVC.from_local(self.model_dir, self.device)
            self.vc_sr = int(self.vc.sr)

    def _generate(self, text, language, params):
        return self.model.generate(text, **self._common(params))

    def tts(self, text, voice, language="en", **params):
        self._ensure_tts()
        return super().tts(text, voice, language, **params)

    def tts_stream(self, text, voice, language="en", **params):
        self._ensure_tts()
        yield from super().tts_stream(text, voice, language, **params)

    def prepare_voice(self, voice: Voice) -> None:
        self._ensure_vc()
        self._vc_voice = None
        self._set_vc_voice(voice)

    def _set_vc_voice(self, voice: Voice) -> None:
        import torch

        key = f"{voice.id}:{voice.updated_at}"
        if self._vc_voice == key:
            return
        if not voice.reference_path.exists():
            raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
        cache = voice.cache_dir / f"{self.spec.id}.vc_ref.pt"
        if cache.exists():
            ref = torch.load(cache, map_location=self.device)
            self.vc.ref_dict = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in ref.items()}
        else:
            self.vc.set_target_voice(str(voice.reference_path))
            torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in self.vc.ref_dict.items()}, cache)
        self._vc_voice = key

    def convert(self, wav: np.ndarray, sr: int, voice: Voice, **params) -> tuple[np.ndarray, int]:
        import torch

        self._ensure_vc()
        self._set_vc_voice(voice)
        from chatterbox.models.s3tokenizer import S3_SR

        x = audio.resample(wav, sr, S3_SR)
        with torch.inference_mode():
            t = torch.from_numpy(x).float().to(self.device)[None, :]
            tokens, _ = self.vc.s3gen.tokenizer(t)
            out, _ = self.vc.s3gen.inference(speech_tokens=tokens, ref_dict=self.vc.ref_dict)
        wav_out = to_numpy(out)
        # Filigrane inaudible (Perth) appliqué par Chatterbox pour signaler l'audio généré
        marker = getattr(self.vc, "watermarker", None)
        if marker is not None:
            wav_out = to_numpy(marker.apply_watermark(wav_out, sample_rate=self.vc_sr))
        return wav_out, self.vc_sr
