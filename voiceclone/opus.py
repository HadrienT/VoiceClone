"""Compression Opus du flux Live (navigateur ↔ serveur) avec PyAV (`pip install av`).

Le navigateur encode/décode avec WebCodecs ; chaque message WebSocket binaire est un paquet Opus brut
de 20 ms. Environ 30 à 40 kbit/s au lieu de ~770 kbit/s en PCM 16 bits à 48 kHz : plus stable à
travers un tunnel SSH ou une connexion lente. Sans PyAV, le Live reste en PCM.
"""

from __future__ import annotations

import fractions
import importlib.util

import numpy as np

OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


def available() -> bool:
    return importlib.util.find_spec("av") is not None


class OpusDecoder:
    def __init__(self, sample_rate: int) -> None:
        import av

        self._av = av
        self.ctx = av.CodecContext.create("opus", "r")
        self.ctx.sample_rate = sample_rate
        self.ctx.layout = "mono"
        self.ctx.open()

    def decode(self, packet: bytes) -> np.ndarray:
        out = [f.to_ndarray().astype(np.float32).reshape(-1) for f in self.ctx.decode(self._av.Packet(packet))]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class OpusEncoder:
    """Encode un flux float32 mono en paquets Opus de 20 ms."""

    def __init__(self, sample_rate: int = 48000, bitrate: int = 48000) -> None:
        import av

        self._av = av
        self.sr = sample_rate
        self.ctx = av.CodecContext.create("libopus", "w")
        self.ctx.sample_rate = sample_rate
        self.ctx.layout = "mono"
        self.ctx.format = "s16"
        self.ctx.bit_rate = bitrate
        self.ctx.time_base = fractions.Fraction(1, sample_rate)
        self.ctx.open()
        self.frame = self.ctx.frame_size or sample_rate // 50
        self._pending = np.zeros(0, dtype=np.int16)
        self._pts = 0

    def encode(self, x: np.ndarray) -> list[bytes]:
        pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
        self._pending = np.concatenate([self._pending, pcm])
        packets = []
        while len(self._pending) >= self.frame:
            chunk, self._pending = self._pending[: self.frame], self._pending[self.frame:]
            f = self._av.AudioFrame.from_ndarray(chunk[None, :], format="s16", layout="mono")
            f.sample_rate = self.sr
            f.pts = self._pts
            self._pts += self.frame
            packets += [bytes(p) for p in self.ctx.encode(f)]
        return packets
