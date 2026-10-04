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


def test_sola_offset_finds_phase():
    from voiceclone.realtime import sola_offset

    sr = 24000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    tail = x[1000:1480]
    seg = x[1000 - 37:]  # la même onde, décalée de 37 échantillons
    assert sola_offset(tail, seg, 120) == 37


def test_browser_live_say_silence_and_ping(client, fake_model):
    v = _voice(client)
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({"mode": "passthrough", "voice_id": v["id"], "say_model_id": "fake",
                                 "sample_rate": 16000}))
        _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "started")
        ws.send_text(json.dumps({"type": "ping", "t": 123}))
        pong = _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "pong")
        assert json.loads(pong["text"])["t"] == 123
        ws.send_text(json.dumps({"type": "silence", "n": 320}))  # silence transmis sans audio
        audio_back = _receive_until(ws, lambda m: m.get("bytes"))
        assert not np.any(np.frombuffer(audio_back["bytes"], dtype="<i2"))
        ws.send_text(json.dumps({"type": "say", "text": "Salut Discord"}))
        loud = _receive_until(ws, lambda m: m.get("bytes") and np.abs(np.frombuffer(m["bytes"], dtype="<i2")).max() > 1000)
        assert loud  # la phrase tapée revient en audio, avant même le statut qui la mentionne
        status = _receive_until(ws, lambda m: m.get("text") and any(
            t.get("typed") for t in json.loads(m["text"]).get("transcripts", [])))
        assert json.loads(status["text"])["transcripts"][-1]["text"] == "Salut Discord"
        ws.send_text("stop")


def test_say_requires_model(client, fake_model):
    v = _voice(client)
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({"mode": "passthrough", "voice_id": v["id"], "sample_rate": 16000}))
        _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "started")
        ws.send_text(json.dumps({"type": "say", "text": "x"}))
        notice = _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "notice")
        assert "modèle" in json.loads(notice["text"])["detail"]
        ws.send_text("stop")


def test_browser_live_opus_roundtrip(client, fake_model):
    import pytest

    from voiceclone import opus

    if not opus.available():
        pytest.skip("PyAV absent")
    v = _voice(client)
    sr = 48000
    enc = opus.OpusEncoder(sr)
    t = np.arange(sr * 2) / sr
    packets = enc.encode((0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({"mode": "passthrough", "voice_id": v["id"], "sample_rate": sr, "codec": "opus"}))
        started = _receive_until(ws, lambda m: m.get("text") and json.loads(m["text"])["type"] == "started")
        assert json.loads(started["text"])["codec"] == "opus"
        for p in packets:
            ws.send_bytes(p)
        dec = opus.OpusDecoder(48000)
        got = []
        while sum(map(len, got)) < sr:  # ~1 s d'audio renvoyé
            m = _receive_until(ws, lambda m: m.get("bytes"))
            assert len(m["bytes"]) < 400  # paquets Opus, pas du PCM
            got.append(dec.decode(m["bytes"]))
        y = np.concatenate(got)
        assert np.abs(y).max() > 0.1
        ws.send_text("stop")
