import numpy as np

from voiceclone import audio
from voiceclone.engines.base import split_text
from tests.conftest import make_voice_wav


def test_load_resample_and_roundtrip():
    x, sr = audio.load_audio(make_voice_wav(2.0, sr=44100), target_sr=24000)
    assert sr == 24000
    assert abs(len(x) / sr - 3.6) < 0.01  # 2 s + 2 x 0.8 s de silence
    y, sr2 = audio.load_audio(audio.to_wav_bytes(x, sr))
    assert sr2 == sr and len(y) == len(x)


def test_trim_silence_removes_padding():
    x, sr = audio.load_audio(make_voice_wav(2.0, sr=24000))
    trimmed = audio.trim_silence(x, sr)
    assert 2.0 <= len(trimmed) / sr < 2.5


def test_analyze_warns_on_short_audio():
    x, sr = audio.load_audio(make_voice_wav(2.0, sr=24000))
    info = audio.analyze(x, sr)
    assert any("6 s" in w for w in info["warnings"])


def test_crossfade_concat_length():
    a = np.ones(1000, dtype=np.float32)
    out = audio.crossfade_concat([a, a], 1000, fade_ms=100)
    assert len(out) == 1900


def test_split_text():
    text = "Bonjour. " * 80
    chunks = split_text(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).count("Bonjour") == 80
    long = "mot " * 200
    assert all(len(c) <= 100 for c in split_text(long, 100))
