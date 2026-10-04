import json

import numpy as np

from tests.conftest import make_voice_wav


def _voice(client):
    files = [("files", ("a.wav", make_voice_wav(8.0), "audio/wav"))]
    return client.post("/api/voices", data={"name": "Live", "consent": "true"}, files=files).json()


def _receive_until(ws, predicate, limit=400):
    for _ in range(limit):
        msg = ws.receive()
        if predicate(msg):
            return msg
    raise AssertionError("message attendu non reçu")


def test_browser_live_vc_roundtrip(client, fake_model):
    v = _voice(client)
    sr = 48000
    t = np.arange(int(sr * 2)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2")
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({"mode": "vc", "model_id": "fake", "voice_id": v["id"], "sample_rate": sr,
                                 "chunk_ms": 500, "context_ms": 200, "silence_db": -60}))
        _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "started")
        block = int(sr * 0.02)
        for i in range(0, len(pcm), block):
            ws.send_bytes(pcm[i:i + block].tobytes())
        audio = _receive_until(ws, lambda m: m.get("bytes"))
        assert len(audio["bytes"]) > 0 and len(audio["bytes"]) % 2 == 0
        status = _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "status")
        assert json.loads(status["text"])["state"] == "running"
        ws.send_text("stop")


def test_browser_live_reports_config_errors(client):
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({"mode": "vc", "model_id": "xtts-v2", "voice_id": "inconnue", "sample_rate": 48000}))
        msg = _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "error")
        assert json.loads(msg["text"])["detail"]
