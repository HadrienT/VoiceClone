import io

import numpy as np
import pytest
import soundfile as sf

from voiceclone import config, registry
from voiceclone.downloads import COMPLETE_MARKER
from voiceclone.registry import ModelSpec


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    config.ensure_dirs()
    yield tmp_path


@pytest.fixture
def fake_model(tmp_data):
    spec = ModelSpec(
        id="fake", name="Fake", description="test", capabilities=("tts", "vc", "asr"),
        engine="tests.fake_engine:FakeEngine", repos=(registry.HFRepo("org/fake"),),
        packages=("numpy",), pip="-", languages=("fr", "en"),
    )
    registry.register_model(spec)
    d = config.MODELS_DIR / "fake"
    d.mkdir(parents=True)
    (d / COMPLETE_MARKER).write_text("{}")
    yield spec
    registry.unregister_model("fake")


def make_voice_wav(seconds=8.0, sr=44100, freq=180.0, noise=0.0) -> bytes:
    """Signal « vocal » synthétique : harmoniques modulées + silences au début/fin."""
    t = np.arange(int(seconds * sr)) / sr
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) * 0.6 + 0.2
    x = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 6)) * env * 0.3
    pad = np.zeros(int(0.8 * sr))
    x = np.concatenate([pad, x, pad]) + noise * np.random.default_rng(0).standard_normal(len(pad) * 2 + len(x))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_data):
    from fastapi.testclient import TestClient

    from voiceclone.server import create_app

    with TestClient(create_app()) as c:
        yield c
