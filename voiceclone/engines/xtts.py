"""Coqui XTTS v2 (paquet `coqui-tts`)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..voices import Voice
from .base import Engine, EngineError, mix_sources, read_audio_without_torchcodec, split_text, to_numpy, weighted

LANG_ALIASES = {"zh": "zh-cn"}


class XTTSEngine(Engine):
    def load(self) -> None:
        import torch
        import transformers.pytorch_utils as hf_utils

        # coqui-tts importe encore ce helper, supprimé dans transformers 5 (version imposée par Chatterbox)
        if not hasattr(hf_utils, "isin_mps_friendly"):
            hf_utils.isin_mps_friendly = torch.isin
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models import xtts as xtts_module
        from TTS.tts.models.xtts import Xtts

        read_audio_without_torchcodec(xtts_module)  # lecture de la référence via soundfile

        cfg = XttsConfig()
        cfg.load_json(str(self.model_dir / "config.json"))
        model = Xtts.init_from_config(cfg)
        model.load_checkpoint(cfg, checkpoint_dir=str(self.model_dir), eval=True)
        model.to(torch.device(self.device))
        self.model = model
        self.sr = int(getattr(cfg.audio, "output_sample_rate", 24000))
        self._latents: dict[str, tuple] = {}

    # ----------------------------------------------------------- conditionnement
    def _conditioning(self, voice: Voice):
        import torch

        key = f"{voice.id}:{voice.updated_at}"
        if key in self._latents:
            return self._latents[key]
        cache = voice.cache_dir / f"{self.spec.id}.pt"
        if cache.exists():
            d = torch.load(cache, map_location=self.device)
            lat = (d["gpt_cond_latent"].to(self.device), d["speaker_embedding"].to(self.device))
        elif mix := mix_sources(voice):  # voix mélangée : moyenne pondérée des empreintes
            lats = [self._conditioning(v) for v, _ in mix]
            ws = [w for _, w in mix]
            gpt, spk = weighted([g for g, _ in lats], ws), weighted([k for _, k in lats], ws)
            torch.save({"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu()}, cache)
            lat = (gpt, spk)
        else:
            if not voice.reference_path.exists():
                raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
            gpt, spk = self.model.get_conditioning_latents(
                audio_path=[str(voice.reference_path)], max_ref_length=30, gpt_cond_len=12,
                gpt_cond_chunk_len=6, sound_norm_refs=False)
            torch.save({"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu()}, cache)
            lat = (gpt, spk)
        self._latents = {key: lat}  # garde seulement la dernière voix en mémoire
        return lat

    def prepare_voice(self, voice: Voice) -> None:
        self._conditioning(voice)

    # ------------------------------------------------------------------ synthèse
    @staticmethod
    def _params(params: dict) -> dict:
        return {
            "temperature": float(params.get("temperature", 0.75)),
            "speed": float(params.get("speed", 1.0)),
            "repetition_penalty": float(params.get("repetition_penalty", 10.0)),
            "top_p": float(params.get("top_p", 0.85)),
            "top_k": int(params.get("top_k", 50)),
            "enable_text_splitting": False,
        }

    def tts(self, text: str, voice: Voice, language: str = "fr", **params) -> tuple[np.ndarray, int]:
        gpt, spk = self._conditioning(voice)
        lang = LANG_ALIASES.get(language, language)
        out = []
        for chunk in split_text(text, 230):
            res = self.model.inference(chunk, lang, gpt, spk, **self._params(params))
            out.append(to_numpy(res["wav"]))
            out.append(np.zeros(int(self.sr * 0.12), dtype=np.float32))
        if not out:
            raise EngineError("Texte vide.")
        return np.concatenate(out[:-1]), self.sr

    def tts_stream(self, text: str, voice: Voice, language: str = "fr", **params) -> Iterator[tuple[np.ndarray, int]]:
        gpt, spk = self._conditioning(voice)
        lang = LANG_ALIASES.get(language, language)
        p = self._params(params)
        p.pop("enable_text_splitting")
        for chunk in split_text(text, 230):
            for piece in self.model.inference_stream(chunk, lang, gpt, spk, stream_chunk_size=20, **p):
                yield to_numpy(piece), self.sr
