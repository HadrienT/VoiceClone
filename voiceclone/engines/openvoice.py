"""OpenVoice V2 - Tone Color Converter (conversion de timbre zero-shot, très rapide)."""

from __future__ import annotations

import numpy as np

from .. import audio
from ..voices import Voice
from .base import Engine, EngineError, to_numpy


class OpenVoiceEngine(Engine):
    def load(self) -> None:
        from openvoice.api import ToneColorConverter

        conv_dir = self.model_dir / "converter"
        try:
            tcc = ToneColorConverter(str(conv_dir / "config.json"), device=self.device, enable_watermark=False)
        except TypeError:  # anciennes versions sans l'argument
            tcc = ToneColorConverter(str(conv_dir / "config.json"), device=self.device)
        tcc.load_ckpt(str(conv_dir / "checkpoint.pth"))
        self.tcc = tcc
        self.hps = tcc.hps
        self.sr = int(tcc.hps.data.sampling_rate)
        self._se: dict[str, object] = {}
        self._src_se = None

    # --------------------------------------------------------------- empreintes
    def _spec(self, x: np.ndarray):
        import torch
        from openvoice.mel_processing import spectrogram_torch

        h = self.hps.data
        y = torch.from_numpy(x).float().to(self.device).unsqueeze(0)
        return spectrogram_torch(y, h.filter_length, h.sampling_rate, h.hop_length, h.win_length, center=False)

    def _embedding(self, x: np.ndarray):
        """Empreinte de timbre (speaker embedding) d'un signal au taux du modèle."""
        import torch

        with torch.no_grad():
            spec = self._spec(x)
            return self.tcc.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)

    def _target_se(self, voice: Voice):
        import torch

        key = f"{voice.id}:{voice.updated_at}"
        if key in self._se:
            return self._se[key]
        cache = voice.cache_dir / f"{self.spec.id}.se.pt"
        if cache.exists():
            se = torch.load(cache, map_location=self.device)
        else:
            if not voice.reference_path.exists():
                raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
            x, _ = voice.reference_audio(self.sr)
            # moyenne sur des fenêtres de 10 s pour une empreinte stable
            win = self.sr * 10
            parts = [x[i:i + win] for i in range(0, len(x), win) if len(x[i:i + win]) > self.sr * 2] or [x]
            se = torch.stack([self._embedding(p) for p in parts]).mean(0)
            torch.save(se.cpu(), cache)
            se = se.to(self.device)
        if len(self._se) >= 4:
            self._se.pop(next(iter(self._se)))
        self._se[key] = se
        return se

    def prepare_voice(self, voice: Voice) -> None:
        self._target_se(voice)

    # --------------------------------------------------------------- conversion
    def convert(self, wav: np.ndarray, sr: int, voice: Voice, **params) -> tuple[np.ndarray, int]:
        import torch

        tgt = self._target_se(voice)
        x = audio.resample(wav, sr, self.sr)
        source_voice: Voice | None = params.get("source_voice")
        if source_voice is not None:
            src = self._target_se(source_voice)
        elif params.get("stream"):
            # En direct : empreinte source lissée au fil des morceaux pour éviter les sauts
            cur = self._embedding(x)
            self._src_se = cur if self._src_se is None else 0.9 * self._src_se + 0.1 * cur
            src = self._src_se
        else:
            src = self._embedding(x)
        tau = float(params.get("tau", 0.3))
        with torch.no_grad():
            spec = self._spec(x)
            lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            out = self.tcc.model.voice_conversion(spec, lengths, sid_src=src, sid_tgt=tgt, tau=tau)[0][0, 0]
        return to_numpy(out), self.sr

    def reset_stream(self) -> None:
        self._src_se = None
