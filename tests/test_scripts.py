import subprocess
import sys
from pathlib import Path

from tests.conftest import make_voice_wav

ROOT = Path(__file__).resolve().parent.parent


def run(tmp_path, *args):
    env = {"VOICECLONE_DATA": str(tmp_path / "data"), "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, str(ROOT / "scripts/add_samples.py"), *args],
                          capture_output=True, text=True, env=env)


def test_add_samples_with_split(tmp_path):
    src = tmp_path / "long.wav"
    src.write_bytes(make_voice_wav(25.0))
    r = run(tmp_path, "--create", "Ma voix", "--consent", str(src), "--split")
    assert r.returncode == 0, r.stderr
    from voiceclone.voices import VoiceStore

    v = VoiceStore(tmp_path / "data" / "voices").list()[0]
    assert v.name == "Ma voix" and len(v.samples) >= 3
    assert all(s.duration <= 11.5 for s in v.samples)
    # ajout à une voix existante, par nom insensible à la casse
    r = run(tmp_path, "ma voix", str(src))
    assert r.returncode == 0 and "1 échantillon(s) ajouté(s)" in r.stdout
    assert run(tmp_path, "inconnue", str(src)).returncode == 1
    assert run(tmp_path, "--create", "X", str(src)).returncode == 2  # --consent obligatoire
