import json
import sys
import types

import numpy as np
import pytest

from voiceclone.engines.base import EngineError, _TorchaudioWithoutCodec, read_audio_without_torchcodec
from voiceclone.engines.f5tts import F5TTSEngine
from voiceclone.registry import get_model
from voiceclone.voices import VoiceStore
from tests.conftest import make_voice_wav


@pytest.fixture
def f5_voice(tmp_data):
    store = VoiceStore()
    v = store.create("F5")
    v = store.add_sample(v.id, make_voice_wav(6.0))
    engine = F5TTSEngine(get_model("f5-tts"), tmp_data / "models" / "f5-tts", "cpu")
    return engine, v


def test_f5_uses_known_transcript(f5_voice):
    engine, v = f5_voice
    ref_file, _ = engine._reference(v)
    assert engine._ref_text(v, ref_file, "  Bonjour à tous.  ") == "Bonjour à tous."


def test_f5_without_whisper_gives_clear_error(f5_voice):
    engine, v = f5_voice
    ref_file, known = engine._reference(v)
    assert known == ""
    with pytest.raises(EngineError, match="texte exact"):
        engine._ref_text(v, ref_file, known)


def test_f5_reuses_cached_transcription(f5_voice, monkeypatch):
    engine, v = f5_voice
    ref_file, _ = engine._reference(v)
    calls = []
    monkeypatch.setattr(engine, "_transcribe", lambda f, lang: calls.append(f) or "texte transcrit")
    assert engine._ref_text(v, ref_file, "") == "texte transcrit"
    assert engine._ref_text(v, ref_file, "") == "texte transcrit"
    assert len(calls) == 1
    assert json.loads((v.cache_dir / "f5_ref_text.json").read_text())["text"] == "texte transcrit"


def test_torchaudio_shim_delegates_and_reads_with_soundfile(tmp_path, monkeypatch):
    # faux torch minimal : seul from_numpy est utilisé par le shim
    fake_torch = types.SimpleNamespace(from_numpy=lambda a: a)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    real = types.SimpleNamespace(load=lambda *a: (_ for _ in ()).throw(OSError("libnvrtc.so.13")),
                                 functional="functional-réel")
    module = types.SimpleNamespace(torchaudio=real)
    read_audio_without_torchcodec(module)
    read_audio_without_torchcodec(module)  # idempotent
    assert isinstance(module.torchaudio, _TorchaudioWithoutCodec)
    assert module.torchaudio._real is real
    assert module.torchaudio.functional == "functional-réel"
    path = tmp_path / "a.wav"
    path.write_bytes(make_voice_wav(1.0, sr=16000))
    x, sr = module.torchaudio.load(path)
    assert sr == 16000 and x.shape[0] == 1 and x.dtype == np.float32


def test_f5_prefers_short_transcribed_sample(tmp_data):
    store = VoiceStore()
    v = store.create("Longue")
    for secs in (14.0, 9.0, 8.0):  # total > 12 s : F5 doit choisir un échantillon court
        v = store.add_sample(v.id, make_voice_wav(secs))
    engine = F5TTSEngine(get_model("f5-tts"), tmp_data / "models" / "f5-tts", "cpu")
    ref, text = engine._reference(v)
    assert ref.endswith("002.wav") and text == ""
    v = store.set_transcripts(v.id, {"samples/003.wav": "troisième"})
    ref, text = engine._reference(v)
    assert ref.endswith("003.wav") and text == "troisième"
