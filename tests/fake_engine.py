"""Moteur factice pour tester l'API sans télécharger de vrais modèles."""

import numpy as np

from voiceclone.engines.base import Engine


class FakeEngine(Engine):
    SR = 16000

    def load(self):
        self.prepared = []

    def prepare_voice(self, voice):
        self.prepared.append(voice.id)

    def tts(self, text, voice, language="fr", **params):
        n = int(self.SR * min(2.0, 0.05 * len(text)))
        t = np.arange(n) / self.SR
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), self.SR

    def convert(self, wav, sr, voice, **params):
        # « conversion » : même durée, rééchantillonné à 22050 Hz pour tester les ratios
        from voiceclone import audio
        return audio.resample(wav, sr, 22050) * 0.5, 22050

    def transcribe(self, wav, sr, language=None):
        return "bonjour tout le monde"
