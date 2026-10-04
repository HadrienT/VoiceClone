from tests.conftest import make_voice_wav


def _voice(client, n=3):
    files = [("files", (f"{i}.wav", make_voice_wav(3.0 + i), "audio/wav")) for i in range(n)]
    return client.post("/api/voices", data={"name": "V", "consent": "true", "consent_owner": "other"}, files=files).json()


def test_consent_is_recorded(client):
    v = _voice(client, 1)
    c = v["consent"]
    assert c["confirmed"] and c["owner"] == "other" and c["via"] == "web" and c["at"] > 0
    assert "autorisation" in c["statement"]


def test_reorder_samples_changes_reference(client):
    v = _voice(client, 3)
    names = [s["file"].split("/")[-1] for s in v["samples"]]
    r = client.put(f"/api/voices/{v['id']}/order", json={"files": [names[2], names[0], names[1]]})
    assert r.status_code == 200
    assert [s["file"].split("/")[-1] for s in r.json()["samples"]] == [names[2], names[0], names[1]]


def test_voice_settings_are_merged(client):
    v = _voice(client, 1)
    client.patch(f"/api/voices/{v['id']}", json={"settings": {"tts": {"model_id": "xtts-v2", "language": "fr"}}})
    r = client.patch(f"/api/voices/{v['id']}", json={"settings": {"live": {"model_id": "openvoice-v2"}}})
    assert r.json()["settings"] == {"tts": {"model_id": "xtts-v2", "language": "fr"}, "live": {"model_id": "openvoice-v2"}}


def test_history_favorite_params_and_mp3(client, fake_model):
    v = _voice(client, 1)
    r = client.post("/api/tts", json={"model_id": "fake", "voice_id": v["id"], "text": "Bonjour", "params": {"speed": 1.2}})
    item_id = r.headers["X-History-Id"]
    h = client.get("/api/history").json()[0]
    assert h["params"] == {"speed": 1.2}
    assert client.patch(f"/api/history/{item_id}", json={"favorite": True}).json()["favorite"] is True
    mp3 = client.get(f"/api/history/{item_id}/audio?format=mp3")
    assert mp3.status_code == 200 and mp3.headers["content-type"] == "audio/mpeg" and len(mp3.content) > 100


def test_favorites_survive_pruning(tmp_data):
    import numpy as np

    from voiceclone import history as hmod

    h = hmod.History()
    first = h.add(np.zeros(100, dtype=np.float32), 16000, text="garde-moi")
    h.update(first["id"], favorite=True)
    old_max, hmod.MAX_ITEMS = hmod.MAX_ITEMS, 3
    try:
        for i in range(5):
            h.add(np.zeros(100, dtype=np.float32), 16000, text=str(i))
    finally:
        hmod.MAX_ITEMS = old_max
    assert any(i["id"] == first["id"] for i in h.list(100))
