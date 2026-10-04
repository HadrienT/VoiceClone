import json
import sys
import textwrap
import time

import pytest

from tests.conftest import make_voice_wav
from voiceclone import config, registry, settings, training
from voiceclone.engines.base import EngineError
from voiceclone.engines.rvc import voice_model


def _voice(client, n=3, transcript="Bonjour, ceci est un essai de ma voix."):
    files = [("files", (f"{i}.wav", make_voice_wav(4.0 + i), "audio/wav")) for i in range(n)]
    v = client.post("/api/voices", data={"name": "Moi", "consent": "true"}, files=files).json()
    texts = {s["file"].split("/")[-1]: transcript for s in v["samples"][:-1]}  # le dernier sans texte
    return client.put(f"/api/voices/{v['id']}/transcripts", json={"texts": texts}).json()


def _wait(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError("tâche trop longue")


def test_xtts_dataset(client):
    v = _voice(client)
    voice = client.app.state.voices.get(v["id"])
    out = config.DATA_DIR / "ds"
    info = training.build_xtts_dataset(voice, out, transcribe=lambda x, sr: "texte reconnu")
    assert info["clips"] == 3 and info["eval"] == 1 and info["train"] == 2
    lines = (out / "metadata_train.csv").read_text().splitlines()
    assert lines[0] == "audio_file|text|speaker_name" and lines[1].startswith("wavs/")
    # sans transcription automatique : seuls les 2 extraits transcrits sont gardés
    assert training.build_xtts_dataset(voice, config.DATA_DIR / "ds2")["clips"] == 2
    v2 = _voice(client, n=2)  # un seul extrait transcrit : pas assez
    with pytest.raises(ValueError, match="Pas assez"):
        training.build_xtts_dataset(client.app.state.voices.get(v2["id"]), config.DATA_DIR / "ds3")


FAKE_XTTS = textwrap.dedent('''
    import json, sys, pathlib
    p = json.loads(sys.argv[-1])
    out = pathlib.Path(p["output"]); out.mkdir(parents=True, exist_ok=True)
    for e in range(p["epochs"]):
        print(f" > EPOCH: {e}/{p['epochs']}", flush=True)
    assert (pathlib.Path(p["dataset"]) / "metadata_train.csv").exists()
    (out / "model.pth").write_bytes(b"x" * 2000)
    (out / "result.json").write_text(json.dumps({"model": str(out / "model.pth")}))
''')


def test_xtts_finetune_flow(client, tmp_path):
    base = config.MODELS_DIR / "xtts-v2"
    base.mkdir(parents=True)
    for f in ("model.pth", "config.json", "vocab.json", "dvae.pth", "mel_stats.pth"):
        (base / f).write_text("{}")
    fake = tmp_path / "fakepython"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'fake_xtts.py'} \"$@\"\n")
    fake.chmod(0o755)
    (tmp_path / "fake_xtts.py").write_text(FAKE_XTTS)
    settings.update(engine_python={"xtts-v2": str(fake)})
    v = _voice(client, transcript="Phrase transcrite.")
    job = client.post("/api/training/xtts", json={"voice_id": v["id"], "epochs": 3}).json()
    j = _wait(client, job["id"])
    assert j["state"] == "done", j
    mid = j["result"]["model_id"]
    assert (config.MODELS_DIR / mid / "model.pth").exists()
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert models[mid]["downloaded"] and "tts" in models[mid]["capabilities"]
    assert client.get(f"/api/voices/{v['id']}").json()["settings"]["tts"]["model_id"] == mid
    assert [m["id"] for m in client.get("/api/training/models").json()] == [mid]
    # persistance : rechargé au démarrage
    registry.unregister_model(mid)
    assert training.load_custom_models() == [mid]
    client.delete(f"/api/training/models/{mid}")
    assert mid not in {m["id"] for m in client.get("/api/models").json()}


FAKE_APPLIO = textwrap.dedent('''
    import os
    def run_prerequisites_script(**k): return "Prerequisites installed successfully."
    def run_preprocess_script(**k):
        assert os.listdir(k["dataset_path"])
        return "Model preprocessed successfully."
    def run_extract_script(**k): return "Model extracted successfully."
    def run_train_script(**k):
        d = os.path.join("logs", k["model_name"]); os.makedirs(d, exist_ok=True)
        for e in range(1, k["total_epoch"] + 1):
            print(f"{k['model_name']} | epoch={e} | step={e*10} | time", flush=True)
        open(os.path.join(d, f"{k['model_name']}_{k['total_epoch']}e_30s.pth"), "wb").write(b"p" * 5000)
        open(os.path.join(d, f"{k['model_name']}.index"), "wb").write(b"i" * 100)
        return "Model trained successfully."
''')


def test_rvc_training_with_applio_and_import(client, tmp_path):
    v = _voice(client)
    assert client.post("/api/training/rvc", json={"voice_id": v["id"]}).status_code == 409  # Applio absent
    applio = tmp_path / "Applio"
    applio.mkdir()
    (applio / "core.py").write_text(FAKE_APPLIO)
    settings.update(applio_dir=str(applio), applio_python=sys.executable)
    job = client.post("/api/training/rvc", json={"voice_id": v["id"], "epochs": 3}).json()
    j = _wait(client, job["id"])
    assert j["state"] == "done", j
    voice = client.app.state.voices.get(v["id"])
    pth, index, version = voice_model(voice)
    assert pth.read_bytes() == b"p" * 5000 and index.exists() and version == "v2"
    # import manuel puis retrait
    r = client.post(f"/api/voices/{v['id']}/rvc", files={"pth": ("m.pth", b"q" * 3000, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["settings"]["rvc"]["source"] == "import : m.pth"
    assert "index" not in r.json()["settings"]["rvc"]
    client.delete(f"/api/voices/{v['id']}/rvc")
    with pytest.raises(EngineError):
        voice_model(client.app.state.voices.get(v["id"]))


def test_seedvc_repo_dir(tmp_path, monkeypatch):
    from voiceclone.engines import seedvc

    repo = tmp_path / "seed"
    (repo / "modules").mkdir(parents=True)
    (repo / "modules" / "commons.py").write_text("")
    monkeypatch.setenv("VOICECLONE_SEEDVC_DIR", str(repo))
    assert seedvc.ensure_repo() == repo and str(repo) in sys.path
    sys.path.remove(str(repo))
    assert json.dumps(registry.get_model("seed-vc").to_dict())


@pytest.mark.parametrize("seed", ["-1", "99999999999", "abc"])
def test_child_process_survives_invalid_hash_seed(monkeypatch, seed):
    import subprocess

    # une bibliothèque du serveur a écrit une graine invalide : le Python enfant doit quand même démarrer
    monkeypatch.setenv("PYTHONHASHSEED", seed)
    assert subprocess.run([sys.executable, "-c", "pass"], env=dict(__import__("os").environ)).returncode != 0
    r = subprocess.run([sys.executable, "-c", "print('ok')"], env=config.child_env(), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok"
    assert config.child_env(X="1")["X"] == "1"
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    assert config.child_env()["PYTHONHASHSEED"] == "42"  # une valeur valide est conservée


def test_xtts_finetune_with_invalid_hash_seed(client, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "-1")
    test_xtts_finetune_flow(client, tmp_path)


def test_patch_coqui_before_tts_import(tmp_path):
    """L'entraînement doit appliquer les mêmes contournements que le moteur XTTS (transformers 5, torchcodec)."""
    import subprocess

    stubs = {
        "torch/__init__.py": "def isin(*a): return 'isin'\n",
        "transformers/__init__.py": "",
        "transformers/pytorch_utils.py": "",  # transformers 5 : plus de isin_mps_friendly
        "TTS/__init__.py": "", "TTS/tts/__init__.py": "", "TTS/tts/models/__init__.py": "",
        "TTS/tts/models/xtts.py": "import torchaudio\n",
        "torchaudio/__init__.py": "def load(p): raise RuntimeError('torchcodec')\n",
        # ce que fait coqui-tts à l'import (tortoise/autoregressive.py)
        "TTS/tts/layers/__init__.py": "",
        "TTS/tts/layers/autoregressive.py": "from transformers.pytorch_utils import isin_mps_friendly as isin\n",
    }
    for rel, code in stubs.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(code)
    code = ("from voiceclone.engines.xtts import patch_coqui\npatch_coqui()\n"
            "import TTS.tts.layers.autoregressive as a\nassert a.isin() == 'isin'\n"
            "import TTS.tts.models.xtts as x\nassert type(x.torchaudio).__name__ == '_TorchaudioWithoutCodec'\n"
            "print('ok')")
    import os

    env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{config.ROOT_DIR}"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
