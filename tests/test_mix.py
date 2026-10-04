import numpy as np

from tests.conftest import make_voice_wav
from voiceclone.engines.base import mix_sources, weighted


def _voice(client, name, freq):
    return client.post("/api/voices", data={"name": name, "consent": "true"},
                       files=[("files", ("a.wav", make_voice_wav(12.0, freq=freq), "audio/wav"))]).json()


def test_mix_voice_creation(client, fake_model):
    a, b = _voice(client, "A", 150), _voice(client, "B", 230)
    r = client.post("/api/voices/mix", json={"name": "AB", "sources": [{"voice_id": a["id"], "weight": 70},
                                                                       {"voice_id": b["id"], "weight": 30}]})
    assert r.status_code == 200, r.text
    m = r.json()
    assert [s["weight"] for s in m["mix"]["sources"]] == [0.7, 0.3]
    assert m["consent"]["via"] == "mix" and set(m["consent"]["sources"]) == {a["id"], b["id"]}
    assert len(m["samples"]) == 2 and m["samples"][0]["original_name"].startswith("A")
    assert m["samples"][0]["duration"] > m["samples"][1]["duration"]
    # utilisable comme n'importe quelle voix
    assert client.post("/api/tts", json={"model_id": "fake", "voice_id": m["id"], "text": "Salut"}).status_code == 200
    voice = client.app.state.voices.get(m["id"])
    srcs = mix_sources(voice)
    assert [(v.id, w) for v, w in srcs] == [(a["id"], 0.7), (b["id"], 0.3)]
    # source supprimée : repli sur la référence assemblée
    client.delete(f"/api/voices/{b['id']}")
    assert mix_sources(client.app.state.voices.get(m["id"])) == []


def test_mix_validation(client):
    a = _voice(client, "A", 150)
    r = client.post("/api/voices/mix", json={"name": "X", "sources": [{"voice_id": a["id"], "weight": 50},
                                                                      {"voice_id": a["id"], "weight": 50}]})
    assert r.status_code == 400


def test_weighted():
    out = weighted([np.ones(3), np.zeros(3)], [0.25, 0.75])
    assert np.allclose(out, 0.25)
