"""Lot 3 et suivants : réglages, diagnostic, journal, file de tâches, LRU, textes longs, filigrane, isolation."""

import io
import time

import numpy as np
import soundfile as sf

from tests.conftest import make_voice_wav
from voiceclone import longform, registry, settings, watermark
from voiceclone.downloads import COMPLETE_MARKER
from voiceclone.jobs import JobManager
from voiceclone.manager import EngineManager, size_gb


def _voice(client):
    files = [("files", ("a.wav", make_voice_wav(4.0), "audio/wav"))]
    return client.post("/api/voices", data={"name": "V", "consent": "true"}, files=files).json()


def _fake(mid, size="≈ 1 Go"):
    from voiceclone import config

    spec = registry.ModelSpec(id=mid, name=mid, description="", capabilities=("tts", "vc", "asr"),
                              engine="tests.fake_engine:FakeEngine", repos=(registry.HFRepo("o/x"),),
                              packages=("numpy",), pip="-", size_hint=size)
    registry.register_model(spec)
    d = config.MODELS_DIR / mid
    d.mkdir(parents=True, exist_ok=True)
    (d / COMPLETE_MARKER).write_text("{}")
    return spec


def test_settings_roundtrip(client):
    assert client.get("/api/settings").json()["watermark"] is False
    r = client.patch("/api/settings", json={"max_loaded_models": 2, "watermark": True})
    assert r.json()["max_loaded_models"] == 2 and r.json()["watermark"] is True
    assert settings.get("max_loaded_models") == 2


def test_diagnostics_and_logs(client):
    d = client.get("/api/diagnostics").json()
    assert d["python"] and any(p["dist"] == "numpy" and p["version"] for p in d["packages"])
    assert {"ok", "label"} <= set(d["checks"][0])
    import logging

    logging.getLogger("voiceclone.test").warning("message de test 123")
    logs = client.get("/api/logs?level=WARNING").json()
    assert any("message de test 123" in r["msg"] for r in logs["records"])
    assert client.get(f"/api/logs?after={logs['last']}&level=WARNING").json()["records"] == []


def test_job_manager_progress_cancel_and_error():
    jm = JobManager()

    def ok(job):
        for i in range(3):
            job.update(i / 3, f"étape {i}")
        return {"x": 1}

    j = jm.wait(jm.submit("t", "ok", ok).id)
    assert j.state == "done" and j.result == {"x": 1} and j.progress == 1.0
    assert jm.wait(jm.submit("t", "ko", lambda job: 1 / 0).id).state == "error"

    def slow(job):
        while True:
            job.update(0.5)
            time.sleep(0.01)

    s = jm.submit("t", "lent", slow)
    time.sleep(0.1)
    jm.cancel(s.id)
    assert jm.wait(s.id).state == "cancelled"


def test_lru_unload_by_count_and_pins(tmp_data):
    for mid in ("m1", "m2", "m3"):
        _fake(mid)
    try:
        settings.update(max_loaded_models=2)
        m = EngineManager()
        m.get("m1")
        m.get("m2")
        m.pin("m1")
        m.get("m3")  # m2 (non épinglé) est déchargé, m1 est protégé par le Live
        assert set(m._engines) == {"m1", "m3"}
        m.unpin("m1")
        m.get("m2")  # désormais m1 est le moins récemment utilisé
        assert set(m._engines) == {"m3", "m2"}
    finally:
        for mid in ("m1", "m2", "m3"):
            registry.unregister_model(mid)


def test_lru_unload_by_vram(tmp_data, monkeypatch):
    for mid, size in (("a", "≈ 3 Go"), ("b", "≈ 3 Go"), ("c", "≈ 500 Mo")):
        _fake(mid, size)
    try:
        m = EngineManager()
        free = {"gb": 10.0}
        monkeypatch.setattr(m, "free_vram_gb", lambda: free["gb"])
        m.get("a")
        free["gb"] = 2.0  # pas assez pour b (3 Go × 1,2)
        orig = m._evict

        def evict(mid, why):
            free["gb"] += 4.0
            return orig(mid, why)

        monkeypatch.setattr(m, "_evict", evict)
        m.get("b")
        assert set(m._engines) == {"b"}
        assert size_gb(registry.get_model("c")) < 1
    finally:
        for mid in ("a", "b", "c"):
            registry.unregister_model(mid)


def test_longform_parse_and_assemble():
    items = longform.parse("Un. Deux ! [pause 1.5s] Trois [pause 200ms] [pause]\n\nQuatre. [pause]")
    assert [i["type"] for i in items] == ["text", "pause", "text", "pause", "text"]
    assert items[1]["seconds"] == 1.5 and items[3]["seconds"] == 0.8  # pauses consécutives fusionnées
    wav = longform.assemble([(items[0], np.ones(100, np.float32)), (items[1], None)] + [(items[2], np.ones(100, np.float32))], 1000)
    assert len(wav) == 100 + 1500 + 100


def test_long_tts_job_and_segment_regeneration(client, fake_model):
    v = _voice(client)
    job = client.post("/api/tts/long", json={"model_id": "fake", "voice_id": v["id"],
                                             "text": "Première phrase. [pause 1s] Deuxième phrase ici."}).json()
    for _ in range(200):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["state"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert j["state"] == "done", j
    hid = j["result"]["history_id"]
    item = client.get(f"/api/history/{hid}").json()
    segs = item["segments"]
    assert [s["type"] for s in segs] == ["text", "pause", "text"] and segs[0]["file"]
    assert client.get(f"/api/history/{hid}/segments/0/audio").status_code == 200
    before = item["duration"]
    r = client.post(f"/api/history/{hid}/segments/2", json={"text": "Ok."})
    assert r.status_code == 200, r.text
    assert r.json()["segments"][2]["text"] == "Ok." and r.json()["duration"] < before
    client.delete(f"/api/history/{hid}")
    assert client.get(f"/api/history/{hid}").status_code == 404


def _harmonic(sr=24000, seconds=8):
    t = np.arange(sr * seconds) / sr
    f0 = 130 + 30 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, 30)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    return (0.2 * x + 0.005 * np.random.default_rng(1).standard_normal(len(t))).astype(np.float32)


def test_watermark_embed_detect_robust():
    x = _harmonic()
    m = watermark.embed(x, 24000)
    snr = 10 * np.log10(np.sum(x ** 2) / np.sum((m - x) ** 2))
    assert snr > 30
    assert watermark.score(x, 24000) < watermark.THRESHOLD
    assert watermark.score(m, 24000) > watermark.THRESHOLD
    assert watermark.score(0.3 * m[12345:], 24000) > watermark.THRESHOLD  # décalé, coupé, moins fort
    for seed in range(5):
        noise = np.random.default_rng(seed).standard_normal(24000 * 6).astype(np.float32) * 0.1
        assert watermark.score(noise, 24000) < watermark.THRESHOLD


def test_watermark_applied_to_outputs_and_detect_endpoint(client, fake_model):
    v = _voice(client)
    client.patch("/api/settings", json={"watermark": True})
    r = client.post("/api/tts", json={"model_id": "fake", "voice_id": v["id"], "text": "x" * 60})
    assert client.get("/api/history").json()[0]["watermark"] is True
    d = client.post("/api/watermark/detect", files={"file": ("a.wav", r.content, "audio/wav")}).json()
    assert d["detected"], d
    buf = io.BytesIO()
    sf.write(buf, _harmonic(), 24000, format="WAV")
    d = client.post("/api/watermark/detect", files={"file": ("b.wav", buf.getvalue(), "audio/wav")}).json()
    assert not d["detected"]


def test_remote_engine_runs_in_subprocess(client, fake_model):
    import sys

    settings.update(engine_python={"fake": sys.executable})
    v = _voice(client)
    r = client.post("/api/tts", json={"model_id": "fake", "voice_id": v["id"], "text": "Bonjour"})
    assert r.status_code == 200, r.text
    eng = client.app.state.manager.get("fake")
    assert type(eng).__name__ == "RemoteEngine" and eng._proc.poll() is None
    assert eng.transcribe(np.zeros(16000, np.float32), 16000) == "bonjour tout le monde"
    chunks = list(eng.tts_stream("salut", client.app.state.voices.get(v["id"])))
    assert len(chunks) == 1 and chunks[0][1] == 16000
    proc = eng._proc
    client.post("/api/models/fake/unload")
    proc.wait(timeout=10)
    assert proc.returncode is not None
