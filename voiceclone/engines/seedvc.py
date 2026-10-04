"""Seed-VC : conversion de voix zero-shot par diffusion (meilleure qualité de timbre qu'OpenVoice).

Seed-VC n'est pas publié sur PyPI : son code (GPL-3.0) est cloné automatiquement depuis GitHub
dans data/repos/seed-vc au premier chargement (ou indiquez VOICECLONE_SEEDVC_DIR), puis ses
modules sont importés. Les poids viennent de Hugging Face via l'onglet Modèles.

L'empreinte de la voix cible (caractéristiques Whisper, mel et style CAMPPlus de la référence) est
calculée une fois puis mise en cache : chaque conversion ne traite plus que l'audio source, ce qui
rend le modèle utilisable en Live avec peu d'étapes de diffusion (4-8).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import audio, config
from ..voices import Voice
from .base import Engine, EngineError, mix_sources, to_numpy, weighted

REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
CHECKPOINT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
CONFIG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
MAX_REF_S = 25


def repo_dir() -> Path:
    return Path(os.environ.get("VOICECLONE_SEEDVC_DIR") or config.DATA_DIR / "repos" / "seed-vc")


def ensure_repo() -> Path:
    d = repo_dir()
    if not (d / "modules" / "commons.py").exists():
        if shutil.which("git") is None:
            raise EngineError(f"Seed-VC : installez git, ou clonez {REPO_URL} dans {d} (ou VOICECLONE_SEEDVC_DIR).")
        d.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(d)], capture_output=True, text=True)
        if r.returncode != 0:
            raise EngineError(f"Impossible de cloner Seed-VC : {r.stderr.strip()[:300]}")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d


class SeedVCEngine(Engine):
    def load(self) -> None:
        ensure_repo()
        import torch
        import yaml
        from modules.audio import mel_spectrogram
        from modules.bigvgan import bigvgan
        from modules.campplus.DTDNN import CAMPPlus
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from transformers import AutoFeatureExtractor, WhisperModel

        dev = torch.device(self.device)
        cfg = yaml.safe_load((self.model_dir / CONFIG).read_text(encoding="utf-8"))
        params = recursive_munch(cfg["model_params"])
        model = build_model(params, stage="DiT")
        model, *_ = load_checkpoint(model, None, str(self.model_dir / CHECKPOINT), load_only_params=True,
                                    ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(dev)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.model = model
        spect = cfg["preprocess_params"]["spect_params"]
        self.sr = int(cfg["preprocess_params"]["sr"])
        self.hop = int(spect["hop_length"])
        mel_args = {"n_fft": spect["n_fft"], "win_size": spect["win_length"], "hop_size": spect["hop_length"],
                    "num_mels": spect["n_mels"], "sampling_rate": self.sr, "fmin": 0, "fmax": None, "center": False}
        self.to_mel = lambda x: mel_spectrogram(x, **mel_args)

        half = dev.type == "cuda"
        wdir = str(self.model_dir / "whisper")
        self.whisper = WhisperModel.from_pretrained(wdir, torch_dtype=torch.float16 if half else torch.float32).to(dev)
        del self.whisper.decoder
        self.features = AutoFeatureExtractor.from_pretrained(wdir)

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(torch.load(self.model_dir / "campplus" / "campplus_cn_common.bin",
                                                 map_location="cpu"))
        self.campplus.eval().to(dev)
        vocoder = bigvgan.BigVGAN.from_pretrained(str(self.model_dir / "bigvgan"), use_cuda_kernel=False)
        vocoder.remove_weight_norm()
        self.vocoder = vocoder.eval().to(dev)
        self.dev = dev
        self._targets: dict[str, tuple] = {}

    # ------------------------------------------------------------ caractéristiques
    def _whisper_features(self, x16):
        """Encodeur Whisper sur l'audio 16 kHz (fenêtres de 30 s, 5 s de recouvrement au-delà)."""
        import torch

        def encode(chunk):
            inputs = self.features([chunk.cpu().numpy()], return_tensors="pt", return_attention_mask=True,
                                   sampling_rate=16000)
            feats = self.whisper._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask)
            out = self.whisper.encoder(feats.to(self.dev).to(self.whisper.encoder.dtype), return_dict=True)
            return out.last_hidden_state.float()[:, : len(chunk) // 320 + 1]

        win, overlap = 16000 * 30, 16000 * 5
        if len(x16) <= win:
            return encode(x16)
        parts, start = [], 0
        while start < len(x16):
            chunk = x16[max(0, start - overlap): start + win - (overlap if start else 0)]
            feats = encode(chunk)
            parts.append(feats if not start else feats[:, overlap // 320:])
            start += win - (overlap if start else 0)
        return torch.cat(parts, dim=1)

    def _style(self, x16):
        import torchaudio

        fbank = torchaudio.compliance.kaldi.fbank(x16[None], num_mel_bins=80, dither=0, sample_frequency=16000)
        return self.campplus((fbank - fbank.mean(dim=0, keepdim=True))[None])

    def _target(self, voice: Voice):
        """(condition de la référence, mel de la référence, style) — calculé une fois par voix."""
        import torch

        key = f"{voice.id}:{voice.updated_at}"
        if key in self._targets:
            return self._targets[key]
        if not voice.reference_path.exists():
            raise EngineError("Cette voix n'a pas encore d'échantillon audio.")
        cache = voice.cache_dir / f"{self.spec.id}.pt"
        if cache.exists():
            d = torch.load(cache, map_location=self.dev)
            target = (d["prompt"], d["mel"], d["style"])
        else:
            x, _ = voice.reference_audio(self.sr)
            with torch.inference_mode():
                ref = torch.from_numpy(x[: self.sr * MAX_REF_S]).float().to(self.dev)
                ref16 = torch.from_numpy(audio.resample(x[: self.sr * MAX_REF_S], self.sr, 16000)).float().to(self.dev)
                mel = self.to_mel(ref[None])
                feats = self._whisper_features(ref16)
                prompt = self.model.length_regulator(feats, ylens=torch.LongTensor([mel.size(2)]).to(self.dev),
                                                     n_quantizers=3, f0=None)[0]
                style = self._style(ref16)
                if mix := mix_sources(voice):  # voix mélangée : style (timbre) moyenné
                    style = weighted([self._target(v)[2] for v, _ in mix], [w for _, w in mix])
            torch.save({"prompt": prompt.cpu(), "mel": mel.cpu(), "style": style.cpu()}, cache)
            target = (prompt, mel, style)
        if len(self._targets) >= 4:
            self._targets.pop(next(iter(self._targets)))
        self._targets[key] = target
        return target

    def prepare_voice(self, voice: Voice) -> None:
        self._target(voice)

    # ------------------------------------------------------------ conversion
    def convert(self, wav: np.ndarray, sr: int, voice: Voice, **params) -> tuple[np.ndarray, int]:
        import torch

        steps = int(params.get("diffusion_steps", 10))
        cfg_rate = float(params.get("inference_cfg_rate", 0.7))
        length = float(params.get("length_adjust", 1.0))
        prompt, mel2, style = self._target(voice)
        x = audio.resample(wav, sr, self.sr)
        if len(x) < self.hop * 4:
            return np.zeros(0, dtype=np.float32), self.sr
        with torch.inference_mode():
            src = torch.from_numpy(x).float().to(self.dev)
            src16 = torch.from_numpy(audio.resample(x, self.sr, 16000)).float().to(self.dev)
            feats = self._whisper_features(src16)
            mel = self.to_mel(src[None])
            n = int(mel.size(2) * length)
            cond = self.model.length_regulator(feats, ylens=torch.LongTensor([n]).to(self.dev), n_quantizers=3,
                                               f0=None)[0]
            # contexte maximal du modèle ≈ 30 s, dont la référence
            window = max(32, self.sr // self.hop * 30 - mel2.size(2))
            pieces = []
            for start in range(0, cond.size(1), window):
                cat = torch.cat([prompt, cond[:, start: start + window]], dim=1)
                with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                    out = self.model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(self.dev), mel2, style,
                                                   None, steps, inference_cfg_rate=cfg_rate)
                out = out[:, :, mel2.size(-1):]
                pieces.append(to_numpy(self.vocoder(out.float())[0]))
        return audio.crossfade_concat(pieces, self.sr, 10.0), self.sr
