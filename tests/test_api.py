import io
import json

import soundfile as sf

from tests.conftest import make_voice_wav


def create_voice(client, name="Test", consent="true", **extra):
    files = [("files", ("a.wav", make_voice_wav(8.0), "audio/wav"))]
    return client.post("/api/voices", data={"name": name, "language": "fr", "consent": consent, **extra}, files=files)


def test_models_list(client):
    ids = {m["id"] for m in client.get("/api/models").json()}
    assert {"xtts-v2", "chatterbox-multilingual", "openvoice-v2", "f5-tts", "whisper-small"} <= ids


def test_model_not_ready(client):
    r = client.post("/api/tts", json={"model_id": "xtts-v2", "voice_id": "x", "text": "a"})
    assert r.status_code in (404, 409)


def test_voice_requires_consent(client):
    assert create_voice(client, consent="false").status_code == 400


def test_voice_lifecycle(client):
    r = create_voice(client, transcript="bonjour")
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["name"] == "Test" and len(v["samples"]) == 1
    assert 7.5 < v["duration"] < 8.8
    assert v["transcript"] == "bonjour"
    assert client.get(f"/api/voices/{v['id']}/audio").status_code == 200

    r = client.post(f"/api/voices/{v['id']}/samples", files={"file": ("b.wav", make_voice_wav(4.0), "audio/wav")})
    v = r.json()
    assert len(v["samples"]) == 2
    assert v["transcript"] == ""  # le second échantillon n'est pas transcrit

    r = client.patch(f"/api/voices/{v['id']}", json={"name": "Renommée"})
    assert r.json()["name"] == "Renommée"

    r = client.delete(f"/api/voices/{v['id']}/samples/002.wav")
    assert len(r.json()["samples"]) == 1
    assert client.delete(f"/api/voices/{v['id']}").status_code == 200
    assert client.get(f"/api/voices/{v['id']}").status_code == 404


def test_invalid_audio_rejected(client):
    files = [("files", ("a.wav", b"pas du son", "audio/wav"))]
    r = client.post("/api/voices", data={"name": "x", "consent": "true"}, files=files)
    assert r.status_code == 400
    assert client.get("/api/voices").json() == []


def test_tts_vc_and_history(client, fake_model):
    v = create_voice(client).json()
    r = client.post("/api/tts", json={"model_id": "fake", "voice_id": v["id"], "text": "Bonjour à tous", "language": "fr"})
    assert r.status_code == 200, r.text
    x, sr = sf.read(io.BytesIO(r.content))
    assert sr == 16000 and len(x) > 0

    r = client.post(f"/api/voices/{v['id']}/prepare", json={"model_id": "fake"})
    assert r.status_code == 200 and "fake" in r.json()["voice"]["prepared"]

    src = make_voice_wav(2.0, sr=16000)
    r = client.post("/api/vc", data={"model_id": "fake", "voice_id": v["id"], "params": json.dumps({})},
                    files={"file": ("s.wav", src, "audio/wav")})
    assert r.status_code == 200, r.text
    _, sr = sf.read(io.BytesIO(r.content))
    assert sr == 22050

    r = client.post("/api/vc", data={"model_id": "fake", "voice_id": v["id"], "mode": "asr_tts", "asr_model_id": "fake"},
                    files={"file": ("s.wav", src, "audio/wav")})
    assert r.status_code == 200
    assert "bonjour" in r.headers["X-Transcript"]

    hist = client.get("/api/history").json()
    assert len(hist) == 3
    assert client.get(f"/api/history/{hist[0]['id']}/audio").status_code == 200

    r = client.post(f"/api/voices/{v['id']}/transcribe", json={"model_id": "fake"})
    assert r.json()["transcript"] == "bonjour tout le monde"

    status = {m["id"]: m for m in client.get("/api/models").json()}["fake"]
    assert status["loaded"] and status["downloaded"]
    client.post("/api/models/fake/unload")
    assert not {m["id"]: m for m in client.get("/api/models").json()}["fake"]["loaded"]


def test_download_flow(client, monkeypatch):
    import time

    import huggingface_hub

    from voiceclone import downloads

    def fake_snapshot(repo_id, local_dir, **kw):
        from pathlib import Path
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "weights.bin").write_bytes(b"0" * 1000)
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(downloads, "_estimate_total", lambda spec: 2000)
    client.post("/api/models/openvoice-v2/download")
    for _ in range(50):
        m = {m["id"]: m for m in client.get("/api/models").json()}["openvoice-v2"]
        if m["download"]["status"] == "done":
            break
        time.sleep(0.05)
    assert m["downloaded"] and m["download"]["progress"] == 1.0
    client.delete("/api/models/openvoice-v2")
    assert not {m["id"]: m for m in client.get("/api/models").json()}["openvoice-v2"]["downloaded"]


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VoiceClone" in r.text


def test_sample_transcripts_and_error_details(client, fake_model, monkeypatch):
    v = create_voice(client).json()
    r = client.put(f"/api/voices/{v['id']}/transcripts", json={"texts": {"001.wav": "Bonjour à tous"}})
    assert r.status_code == 200
    v = r.json()
    assert v["samples"][0]["transcript"] == "Bonjour à tous" and v["transcript"] == "Bonjour à tous"

    # une erreur inattendue doit remonter son vrai message à l'interface
    from tests.fake_engine import FakeEngine

    def boom(self, *a, **k):
        raise RuntimeError("panne de test")

    monkeypatch.setattr(FakeEngine, "tts", boom)
    from fastapi.testclient import TestClient

    from voiceclone.server import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        r = c.post("/api/tts", json={"model_id": "fake", "voice_id": v["id"], "text": "x"})
    assert r.status_code == 500 and "panne de test" in r.json()["detail"]
