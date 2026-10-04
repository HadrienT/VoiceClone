import time

import numpy as np

from tests.conftest import make_voice_wav
from voiceclone import registry
from voiceclone.downloads import COMPLETE_MARKER
from voiceclone.engines.base import EngineError
from voiceclone.manager import EngineManager
from voiceclone.translation import pick_model, speech_to_translated_text, translate_text


def _mt(mid, langs, engine="translate:NLLBEngine"):
    from voiceclone import config

    spec = registry.ModelSpec(id=mid, name=mid, description="", capabilities=("mt",), engine=engine,
                              repos=(registry.HFRepo("o/x"),), packages=("numpy",), pip="-", languages=langs)
    registry.register_model(spec)
    d = config.MODELS_DIR / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / COMPLETE_MARKER).write_text("{}")
    return spec


def test_pick_model_prefers_pair_specific(tmp_data, monkeypatch):
    _mt("any", ("*",))
    _mt("fren", ("fr", "en"), "translate:MarianEngine")
    try:
        assert pick_model("fr", "en") == "fren"
        assert pick_model("en", "fr") == "any"
    finally:
        registry.unregister_model("any")
        registry.unregister_model("fren")


def test_translate_text_and_whisper_fallback(tmp_data, fake_model, monkeypatch):
    m = EngineManager()
    assert translate_text(m, "bonjour", "fr", "fr") == "bonjour"
    try:
        translate_text(m, "bonjour", "fr", "es")
        raise AssertionError("aurait dû échouer")
    except EngineError as exc:
        assert "NLLB" in str(exc)
    asr = m.get("fake")
    src, out = speech_to_translated_text(m, asr, np.zeros(16000, np.float32), 16000, "fr", "en")
    assert out == "[en] bonjour tout le monde"  # modèle de traduction prêt : utilisé en priorité
    # vers l'anglais sans modèle de traduction : Whisper traduit lui-même
    import voiceclone.translation as tr

    monkeypatch.setattr(tr, "pick_model", lambda src, tgt: None)
    src, out = speech_to_translated_text(m, asr, np.zeros(16000, np.float32), 16000, "fr", "en")
    assert (src, out) == ("bonjour tout le monde", "hello everyone")
    # avec un modèle de traduction explicite
    src, out = speech_to_translated_text(m, asr, np.zeros(16000, np.float32), 16000, "fr", "es", "fake")
    assert out == "[es] bonjour tout le monde"


def test_vc_translate_endpoint(client, fake_model):
    v = client.post("/api/voices", data={"name": "V", "consent": "true"},
                    files=[("files", ("a.wav", make_voice_wav(4.0), "audio/wav"))]).json()
    r = client.post("/api/vc", data={"model_id": "fake", "voice_id": v["id"], "mode": "translate",
                                     "asr_model_id": "fake", "language": "fr", "target_language": "de",
                                     "mt_model_id": "fake"},
                    files={"file": ("s.wav", make_voice_wav(2.0), "audio/wav")})
    assert r.status_code == 200, r.text
    from urllib.parse import unquote

    assert unquote(r.headers["X-Translation"]) == "[de] bonjour tout le monde"
    h = client.get("/api/history").json()[0]
    assert h["kind"] == "translate" and h["language"] == "de" and h["source_text"] == "bonjour tout le monde"


def test_live_translation(client, fake_model):
    v = client.post("/api/voices", data={"name": "V", "consent": "true"},
                    files=[("files", ("a.wav", make_voice_wav(4.0), "audio/wav"))]).json()
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_json({"sample_rate": 16000, "mode": "asr_tts", "model_id": "fake", "voice_id": v["id"],
                      "asr_model_id": "fake", "language": "fr", "translate_to": "en", "mt_model_id": "fake",
                      "end_silence_ms": 300, "warmup": False})
        msg = ws.receive_json()
        while msg["type"] == "loading":
            msg = ws.receive_json()
        assert msg["type"] == "started", msg
        sr = 16000
        t = np.arange(sr) / sr
        speech = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        for block in [speech, np.zeros(sr, np.float32)]:
            for i in range(0, len(block), 1600):
                ws.send_bytes((block[i:i + 1600] * 32767).astype("<i2").tobytes())
        deadline = time.time() + 10
        got = None
        while time.time() < deadline:
            m = ws.receive()
            if m.get("text"):
                import json

                d = json.loads(m["text"])
                if d.get("type") == "status" and any("translation" in x for x in d.get("transcripts", [])):
                    got = d["transcripts"][-1]
                    break
        assert got and got["translation"] == "[en] bonjour tout le monde", got
        ws.send_text("stop")
