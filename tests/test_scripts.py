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


def test_smoke_test_script_with_fake_model(tmp_path):
    from voiceclone.downloads import COMPLETE_MARKER

    data = tmp_path / "data"
    (data / "models" / "fake").mkdir(parents=True)
    (data / "models" / "fake" / COMPLETE_MARKER).write_text("{}")
    src = tmp_path / "v.wav"
    src.write_bytes(make_voice_wav(8.0))
    assert run(tmp_path, "--create", "Test", "--consent", str(src)).returncode == 0
    # petit lanceur qui enregistre le moteur factice puis exécute le script
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from voiceclone import registry\n"
        "registry.register_model(registry.ModelSpec(id='fake', name='Fake', description='', "
        "capabilities=('tts','vc','asr'), engine='tests.fake_engine:FakeEngine', "
        "repos=(registry.HFRepo('o/x'),), packages=('numpy',), pip='-', languages=('fr',)))\n"
        f"sys.argv = ['smoke_test.py', '--models', 'fake', '--out', {str(tmp_path / 'out')!r}]\n"
        f"runpy.run_path({str(ROOT / 'scripts/smoke_test.py')!r}, run_name='__main__')\n")
    env = {"VOICECLONE_DATA": str(data), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, str(launcher)], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "4/4 vérifications réussies" in r.stdout
    assert (tmp_path / "out" / "fake-tts.wav").exists()



def test_install_script_dry_run(tmp_path):
    env = {"VOICECLONE_DATA": str(tmp_path / "data"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/install.py"), "--dry-run", "--torch", "cu124",
                        "openvoice-v2", "xtts-v2"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "--no-deps" in r.stdout and "coqui-tts" in r.stdout and "whl/cu124" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts/install.py"), "--list"], capture_output=True,
                       text=True, env=env)
    assert "xtts-v2" in r.stdout
