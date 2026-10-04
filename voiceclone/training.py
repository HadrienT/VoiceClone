"""Entraînement : fine-tuning de XTTS sur une voix et entraînement de modèles RVC via Applio.

Les entraînements tournent dans un processus séparé (mémoire GPU rendue à la fin, annulation
possible) lancé depuis la file de tâches ; leur sortie est copiée dans le journal du serveur.

- XTTS : jeu de données construit à partir des échantillons transcrits de la voix (Whisper complète
  les transcriptions manquantes), entraînement du GPT de XTTS (recette officielle de coqui-tts),
  puis le résultat est ajouté au catalogue comme un nouveau modèle « XTTS · <voix> ».
- RVC : les échantillons sont confiés à Applio (dépôt https://github.com/IAHispano/Applio installé
  à part, avec son propre environnement Python) ; le .pth et l'.index obtenus sont attachés à la voix.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import audio, config, registry, settings
from .downloads import COMPLETE_MARKER
from .jobs import Job, JobCancelled
from .voices import Voice, VoiceStore

log = logging.getLogger(__name__)

XTTS_SR = 22050
MAX_CLIP_S = 11.0  # XTTS : max_wav_length ≈ 11,6 s


def training_dir(kind: str, name: str) -> Path:
    d = config.DATA_DIR / "training" / f"{kind}-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------ modèles personnalisés
def _custom_file() -> Path:
    return config.DATA_DIR / "custom_models.json"


def load_custom_models() -> list[str]:
    """Ajoute au catalogue les modèles issus d'un entraînement (appelé au démarrage)."""
    try:
        items = json.loads(_custom_file().read_text(encoding="utf-8"))
    except Exception:
        return []
    ids = []
    for d in items:
        try:
            registry.register_model(spec_from_dict(d))
            ids.append(d["id"])
        except Exception as exc:
            log.warning("Modèle personnalisé ignoré (%s) : %s", d.get("id"), exc)
    return ids


def spec_from_dict(d: dict) -> registry.ModelSpec:
    base = registry.get_model(d["base"])
    return registry.ModelSpec(
        id=d["id"], name=d["name"], description=d.get("description", ""), capabilities=base.capabilities,
        engine=base.engine, repos=(), packages=base.packages, pip=base.pip, languages=base.languages,
        license=base.license, size_hint=d.get("size_hint", base.size_hint), realtime=base.realtime,
        params=base.params, notes=d.get("notes", "Modèle entraîné localement."))


def save_custom_model(entry: dict) -> registry.ModelSpec:
    try:
        items = json.loads(_custom_file().read_text(encoding="utf-8"))
    except Exception:
        items = []
    items = [i for i in items if i["id"] != entry["id"]] + [entry]
    _custom_file().write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    spec = spec_from_dict(entry)
    registry.register_model(spec)
    return spec


def delete_custom_model(model_id: str) -> None:
    try:
        items = json.loads(_custom_file().read_text(encoding="utf-8"))
    except Exception:
        items = []
    if not any(i["id"] == model_id for i in items):
        raise KeyError(f"Modèle personnalisé introuvable : {model_id}")
    _custom_file().write_text(json.dumps([i for i in items if i["id"] != model_id], ensure_ascii=False, indent=1),
                              encoding="utf-8")
    registry.unregister_model(model_id)
    shutil.rmtree(config.MODELS_DIR / model_id, ignore_errors=True)


def list_custom_models() -> list[dict]:
    try:
        return json.loads(_custom_file().read_text(encoding="utf-8"))
    except Exception:
        return []


# ------------------------------------------------------------ sous-processus
def run_process(job: Job, cmd: list[str], cwd: Path | None = None, env: dict | None = None,
                progress=None, log_path: Path | None = None, companions: list[dict] | None = None) -> None:
    """Lance cmd, recopie sa sortie dans le journal, met à jour la progression, s'arrête si annulé.

    companions : environnements de processus identiques lancés en parallèle (rangs > 0 d'un
    entraînement multi-GPU) ; leur sortie va dans <log_path>.rang<N>.
    """
    log.info("Entraînement : %s", " ".join(str(c) for c in cmd))
    others = []
    for k, cenv in enumerate(companions or [], 1):
        clog = open(f"{log_path or os.devnull}.rang{k}" if log_path else os.devnull, "a", encoding="utf-8")
        others.append((subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                                        env=config.child_env(**{**(env or {}), **cenv}), stdout=clog,
                                        stderr=subprocess.STDOUT, text=True), clog))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=config.child_env(**(env or {})),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    out = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if out:
                out.write(line + "\n")
            if line:
                tail = (tail + [line])[-15:]
                log.info("[entraînement] %s", line[:500])
                if progress:
                    progress(line)
            if job.cancelled:
                proc.terminate()
                break
        proc.wait()
        failed = proc.returncode != 0
        for p, _ in others:
            if failed or job.cancelled:
                p.terminate()
            try:
                p.wait(timeout=None if not (failed or job.cancelled) else 30)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        if out:
            out.close()
        for p, clog in others:
            if p.poll() is None:
                p.kill()
            clog.close()
        if proc.poll() is None:
            proc.kill()
    if job.cancelled:
        raise JobCancelled()
    bad = [k for k, (p, _) in enumerate(others, 1) if p.returncode not in (0, None)]
    if proc.returncode != 0 or bad:
        extra = f"\n(processus GPU {bad} en échec, voir {log_path}.rang*)" if bad else ""
        raise RuntimeError("L'entraînement a échoué :\n" + "\n".join(tail[-6:]) + extra)


# ------------------------------------------------------------ XTTS
def check_training_python(python: str) -> None:
    """Vérifie que le Python d'entraînement importe coqui-tts avant de lancer des heures de calcul."""
    code = ("import numpy, matplotlib, trainer\n"
            "from voiceclone.engines.xtts import patch_coqui\npatch_coqui()\n"
            "import TTS.tts.layers.xtts.trainer.gpt_trainer\nprint(numpy.__version__)")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True, cwd=config.ROOT_DIR,
                       env=config.child_env(PYTHONPATH=str(config.ROOT_DIR)))
    if r.returncode == 0:
        return
    err = (r.stderr.strip().splitlines() or ["?"])[-1]
    if "numpy" in err.lower():
        raise RuntimeError(
            f"Version de numpy incompatible dans l'environnement Python ({err}).\n"
            "Cause probable : rvc-python (ou un autre moteur) installé dans le même environnement a "
            "rétrogradé numpy. Réparez avec :\n"
            f"  {python} -m pip install 'numpy==1.26.4'   (ou : uv pip install 'numpy==1.26.4')\n"
            "puis réinstallez RVC dans un environnement isolé : python scripts/install.py --isolated rvc")
    raise RuntimeError(f"L'environnement d'entraînement XTTS est incomplet : {err}\n"
                       "Installez : pip install coqui-tts (voir Diagnostic → Versions).")


def training_gpus(mode: str = "all", dev: str | None = None) -> list[str]:
    """GPU physiques à utiliser : tous ceux que voit le serveur (« all ») ou seulement le sien (« one »)."""
    dev = dev or manager_device()
    if not dev.startswith("cuda"):
        return []
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if mode == "all":
        n = cuda_count()
        if n > 1:
            return visible[:n] if len(visible) >= n else [str(i) for i in range(n)]
    return [single_gpu_env(dev)["CUDA_VISIBLE_DEVICES"]]


def cuda_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_xtts_dataset(voice: Voice, out: Path, transcribe=None, language: str | None = None) -> dict:
    """Écrit wavs/*.wav (22,05 kHz) + metadata_train.csv / metadata_eval.csv (format « coqui »)."""
    from .prep import split

    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows = []
    learned: dict[str, str] = {}  # transcriptions obtenues pour l'audio d'entraînement (mises en cache)
    for t in voice.training:  # audio d'entraînement : déjà en phrases de 3 à 11 s
        x, _ = audio.load_audio(voice.dir / t["file"], target_sr=XTTS_SR)
        text = t.get("transcript") or ""
        if not text and transcribe is not None:
            text = learned[t["file"]] = transcribe(x, XTTS_SR)
        text = re.sub(r"\s+", " ", (text or "").replace("|", " ")).strip()
        if not text or len(x) < XTTS_SR:
            continue
        name = f"t_{Path(t['file']).stem}.wav"
        audio.save_wav(wavs / name, x, XTTS_SR)
        rows.append((f"wavs/{name}", text, len(x) / XTTS_SR))
    for s in voice.samples:
        x, sr = audio.load_audio(voice.dir / s.file, target_sr=XTTS_SR)
        clips = [x] if len(x) <= MAX_CLIP_S * XTTS_SR else split(x, XTTS_SR, 3.0, MAX_CLIP_S)
        for k, clip in enumerate(clips):
            if len(clip) < XTTS_SR * 1.0:
                continue
            text = s.transcript if len(clips) == 1 else ""
            if not text and transcribe is not None:
                text = transcribe(clip, XTTS_SR)
            text = re.sub(r"\s+", " ", (text or "").replace("|", " ")).strip()
            if not text:
                continue
            name = f"{Path(s.file).stem}_{k:02d}.wav"
            audio.save_wav(wavs / name, clip, XTTS_SR)
            rows.append((f"wavs/{name}", text, len(clip) / XTTS_SR))
    if len(rows) < 2:
        raise ValueError("Pas assez d'audio transcrit : il faut au moins 2 extraits avec leur texte. Ajoutez de "
                         "l'audio d'entraînement et choisissez un modèle Whisper pour le transcrire.")
    n_eval = max(1, round(len(rows) * 0.15))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))
    eval_ids = set(order[:n_eval].tolist())
    header = "audio_file|text|speaker_name\n"
    for fname, keep in (("metadata_train.csv", lambda i: i not in eval_ids), ("metadata_eval.csv", lambda i: i in eval_ids)):
        lines = [f"{r[0]}|{r[1]}|{voice.id}" for i, r in enumerate(rows) if keep(i)]
        (out / fname).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    total = sum(r[2] for r in rows)
    return {"clips": len(rows), "seconds": round(total, 1), "train": len(rows) - n_eval, "eval": n_eval,
            "language": language or voice.language, "learned": learned}


def finetune_xtts(job: Job, manager, voices: VoiceStore, voice_id: str, epochs: int = 10, batch_size: int = 2,
                  grad_accum: int = 4, asr_model_id: str | None = None, language: str | None = None,
                  name: str | None = None, gpus: str = "all", precision: str = "auto") -> dict:
    voice = voices.get(voice_id)
    base_dir = config.MODELS_DIR / "xtts-v2"
    for f in ("model.pth", "config.json", "vocab.json", "dvae.pth", "mel_stats.pth"):
        if not (base_dir / f).exists():
            raise ValueError("Téléchargez d'abord XTTS v2 dans l'onglet Modèles (fichier manquant : " + f + ").")
    python = (settings.get("engine_python") or {}).get("xtts-v2") or sys.executable
    job.update(0.01, "Vérification de l'environnement d'entraînement…")
    check_training_python(python)
    work = training_dir("xtts", voice.id)
    job.update(0.02, "Préparation du jeu de données…")
    transcribe = None
    if asr_model_id:
        asr = manager.get(asr_model_id, "asr")

        def transcribe(x, sr):
            job.check()
            with manager.infer_lock:
                return asr.transcribe(x, sr, voice.language)
    info = build_xtts_dataset(voice, work / "dataset", transcribe, language)
    learned = info.pop("learned")
    if learned:  # transcriptions gardées : le prochain entraînement ne refait pas ce travail
        voices.set_training_transcripts(voice.id, learned)
    job.update(0.05, f"Jeu de données : {info['clips']} extraits, {info['seconds']} s. Entraînement…")
    # libère le GPU pour l'entraînement
    manager.unload_all()
    devices = training_gpus(gpus)
    params = {"dataset": str(work / "dataset"), "output": str(work / "run"), "base_dir": str(base_dir),
              "language": info["language"], "epochs": int(epochs), "batch_size": int(batch_size),
              "grad_accum": int(grad_accum), "precision": precision,
              "workers": max(2, min(8, (os.cpu_count() or 4) // max(1, len(devices)) - 1))}

    def launch(devs: list[str]) -> None:
        label = f"{len(devs)} GPU" if len(devs) > 1 else "1 GPU"

        def progress(line: str) -> None:
            m = re.search(r"EPOCH:\s*(\d+)\s*/\s*(\d+)", line)
            if m:
                done, total = int(m.group(1)), max(1, int(m.group(2)))
                job.update(0.05 + 0.9 * done / total, f"Époque {done + 1}/{total} · {label}")

        p = dict(params, world_size=len(devs), dist_url=f"tcp://127.0.0.1:{free_port()}",
                 group_id=f"group_{work.name}")
        env = {"PYTHONPATH": str(config.ROOT_DIR), "CUDA_VISIBLE_DEVICES": ",".join(devs)} if devs else \
            {"PYTHONPATH": str(config.ROOT_DIR)}
        cmd = [python, "-m", "voiceclone.xtts_train", json.dumps(p)]
        job.update(0.05, f"Entraînement sur {label}…")
        run_process(job, cmd, cwd=config.ROOT_DIR, env={**env, "RANK": "0"}, progress=progress,
                    log_path=work / "train.log", companions=[{"RANK": str(r)} for r in range(1, len(devs))])

    try:
        launch(devices)
    except RuntimeError as exc:
        if len(devices) < 2 or job.cancelled:
            raise
        # le multi-GPU (DDP) peut échouer selon les versions : on retente sur un seul GPU
        log.warning("Entraînement multi-GPU impossible (%s) : nouvel essai sur un seul GPU.", exc)
        shutil.rmtree(work / "run", ignore_errors=True)
        launch(training_gpus("one"))
    result = json.loads((work / "run" / "result.json").read_text(encoding="utf-8"))
    job.update(0.97, "Enregistrement du modèle…")
    model_id = f"xtts-ft-{voice.id}"[:60]
    dest = config.MODELS_DIR / model_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.move(result["model"], dest / "model.pth")
    for f in ("config.json", "vocab.json", "speakers_xtts.pth"):
        if (base_dir / f).exists():
            shutil.copy2(base_dir / f, dest / f)
    (dest / COMPLETE_MARKER).write_text(json.dumps({"trained_at": time.time()}), encoding="utf-8")
    spec = save_custom_model({
        "id": model_id, "base": "xtts-v2", "name": name or f"XTTS · {voice.name}", "voice_id": voice.id,
        "description": f"XTTS v2 affiné sur la voix « {voice.name} » ({info['seconds']} s d'audio, {epochs} époques).",
        "created_at": time.time(), "dataset": info, "size_hint": "≈ 1.8 Go"})
    shutil.rmtree(work / "run", ignore_errors=True)  # points de contrôle intermédiaires (plusieurs Go)
    voices.update(voice.id, settings={"tts": {**(voice.settings.get("tts") or {}), "model_id": spec.id}})
    return {"model_id": spec.id, "dataset": info}


# ------------------------------------------------------------ RVC (Applio)
APPLIO_DRIVER = r'''
import json, sys
sys.path.insert(0, ".")
import core
a = json.loads(sys.argv[1])

def check(r):
    print(r, flush=True)
    if "fail" in str(r).lower() or "error" in str(r).lower():
        sys.exit(1)

print("ÉTAPE 1/4 : modèles pré-entraînés", flush=True)
check(core.run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=False))
print("ÉTAPE 2/4 : prétraitement", flush=True)
check(core.run_preprocess_script(model_name=a["name"], dataset_path=a["dataset"], sample_rate=a["sr"],
      cpu_cores=a["cpu"], cut_preprocess="Automatic", process_effects=False, noise_reduction=False,
      clean_strength=0.7, chunk_len=3.0, overlap_len=0.3, normalization_mode="none"))
print("ÉTAPE 3/4 : extraction des caractéristiques", flush=True)
check(core.run_extract_script(model_name=a["name"], f0_method="rmvpe", cpu_cores=a["cpu"], gpu=a["gpu"],
      sample_rate=a["sr"], embedder_model="contentvec", embedder_model_custom=None, include_mutes=2))
print("ÉTAPE 4/4 : entraînement", flush=True)
check(core.run_train_script(model_name=a["name"], save_every_epoch=a["save_every"], save_only_latest=True,
      save_every_weights=True, total_epoch=a["epochs"], sample_rate=a["sr"], batch_size=a["batch"], gpu=a["gpu"],
      pretrained=True, cleanup=False, index_algorithm="Auto", cache_data_in_gpu=False, custom_pretrained=False,
      vocoder="HiFi-GAN"))
'''


def applio_paths() -> tuple[Path, str]:
    d = Path(settings.get("applio_dir") or config.DATA_DIR / "repos" / "Applio")
    if not (d / "core.py").exists():
        raise ValueError(f"Applio introuvable dans {d}. Installez-le (git clone https://github.com/IAHispano/Applio "
                         "puis son script d'installation) et indiquez son dossier dans Diagnostic → Réglages.")
    py = settings.get("applio_python")
    if not py:
        for cand in (d / ".venv" / "bin" / "python", d / "env" / "bin" / "python", d / ".venv" / "Scripts" / "python.exe",
                     d / "env" / "python.exe"):
            if cand.exists():
                py = str(cand)
                break
    return d, py or sys.executable


def train_rvc(job: Job, manager, voices: VoiceStore, voice_id: str, epochs: int = 200, batch_size: int = 8,
              sample_rate: int = 40000, save_every: int = 25) -> dict:
    voice = voices.get(voice_id)
    if not voice.samples and not voice.training:
        raise ValueError("Cette voix n'a pas d'audio : ajoutez de l'audio d'entraînement.")
    applio, python = applio_paths()
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", voice.id)[:40]
    work = training_dir("rvc", voice.id)
    dataset = work / "dataset"
    dataset.mkdir()
    for s in voice.samples:
        shutil.copy2(voice.dir / s.file, dataset / Path(s.file).name)
    for t in voice.training:
        shutil.copy2(voice.dir / t["file"], dataset / f"t_{Path(t['file']).name}")
    manager.unload_all()
    dev = manager_device()
    gpu = (dev.partition(":")[2] or "0") if dev.startswith("cuda") else "-"
    args = {"name": name, "dataset": str(dataset), "sr": int(sample_rate), "cpu": max(1, (os.cpu_count() or 2) - 1),
            "gpu": gpu, "epochs": int(epochs), "batch": int(batch_size), "save_every": int(save_every)}
    steps = {"ÉTAPE 1/4": 0.02, "ÉTAPE 2/4": 0.05, "ÉTAPE 3/4": 0.10, "ÉTAPE 4/4": 0.15}

    def progress(line: str) -> None:
        for k, v in steps.items():
            if line.startswith(k):
                job.update(v, line)
        m = re.search(r"epoch=(\d+)", line)
        if m:
            e = int(m.group(1))
            job.update(0.15 + 0.83 * min(e, epochs) / epochs, f"Époque {e}/{epochs}")

    driver = work / "applio_driver.py"
    driver.write_text(APPLIO_DRIVER, encoding="utf-8")
    run_process(job, [python, str(driver), json.dumps(args)], cwd=applio, progress=progress,
                log_path=work / "train.log")
    logs = applio / "logs" / name
    weights = sorted(logs.glob(f"{name}_*e_*s.pth"), key=lambda p: p.stat().st_mtime)
    if not weights:
        raise RuntimeError(f"Applio n'a produit aucun modèle dans {logs}.")
    index = next(iter(sorted(logs.glob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True)), None)
    attach_rvc(voices, voice.id, weights[-1].read_bytes(), index.read_bytes() if index else None,
               source=f"Applio ({epochs} époques)")
    return {"voice_id": voice.id, "model": weights[-1].name, "index": index.name if index else None}


def manager_device() -> str:
    from . import device

    return device.get_device()


def single_gpu_env(dev: str | None = None) -> dict:
    """CUDA_VISIBLE_DEVICES limité au GPU utilisé par VoiceClone (ex. "cuda:1" → "1").

    Le Trainer de coqui refuse de démarrer quand plusieurs GPU sont visibles ; dans le processus
    d'entraînement, ce GPU devient alors « cuda:0 ».
    """
    dev = dev or manager_device()
    if not dev.startswith("cuda"):
        return {}
    idx = int(dev.partition(":")[2] or 0)
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return {"CUDA_VISIBLE_DEVICES": visible[idx] if idx < len(visible) else str(idx)}


def attach_rvc(voices: VoiceStore, voice_id: str, pth: bytes, index: bytes | None = None, version: str = "v2",
               source: str = "import") -> Voice:
    """Attache un modèle RVC (.pth + .index facultatif) à une voix."""
    voice = voices.get(voice_id)
    if len(pth) < 1000:
        raise ValueError("Fichier .pth invalide.")
    d = voice.dir / "rvc"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "model.pth").write_bytes(pth)
    entry = {"pth": "rvc/model.pth", "version": version, "source": source, "at": time.time()}
    if index:
        (d / "model.index").write_bytes(index)
        entry["index"] = "rvc/model.index"
    return voices.update(voice_id, settings={"rvc": entry})


def detach_rvc(voices: VoiceStore, voice_id: str) -> Voice:
    voice = voices.get(voice_id)
    shutil.rmtree(voice.dir / "rvc", ignore_errors=True)
    return voices.update(voice_id, settings={"rvc": {}})
