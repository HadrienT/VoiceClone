"""Profils de voix : échantillons importés / enregistrés, nettoyés et prêts pour le clonage.

Arborescence d'une voix :
    data/voices/<id>/
        meta.json
        samples/<n>.wav      échantillons nettoyés (mono 24 kHz)
        reference.wav        référence assemblée (≤ MAX_REFERENCE_SECONDS)
        cache/<model>.*      conditionnements pré-calculés par modèle ("entraînement")
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, config

STORE_SR = 24000

CONSENT_STATEMENTS = {
    "self": "Cette voix est la mienne.",
    "other": "J'ai l'autorisation explicite de la personne dont c'est la voix.",
}


def consent_record(owner: str = "self", via: str = "web") -> dict:
    """Trace horodatée de la déclaration de consentement faite à la création de la voix."""
    owner = owner if owner in CONSENT_STATEMENTS else "self"
    return {"confirmed": True, "at": time.time(), "owner": owner,
            "statement": CONSENT_STATEMENTS[owner], "via": via}


@dataclass
class Sample:
    file: str
    duration: float
    source: str  # "upload" | "record"
    original_name: str = ""
    transcript: str = ""


@dataclass
class Voice:
    id: str
    name: str
    language: str = "fr"
    description: str = ""
    transcript: str = ""  # transcription de reference.wav (utile pour F5-TTS)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    samples: list[Sample] = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    prepared: dict = field(default_factory=dict)  # model_id -> timestamp
    settings: dict = field(default_factory=dict)  # réglages préférés : {"tts": {model_id, language, params}}
    consent: dict = field(default_factory=dict)  # trace du consentement : {confirmed, at, statement, source}

    @property
    def dir(self) -> Path:
        return config.VOICES_DIR / self.id

    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.wav"

    @property
    def cache_dir(self) -> Path:
        d = self.dir / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def reference_audio(self, sr: int | None = None) -> tuple[np.ndarray, int]:
        return audio.load_audio(self.reference_path, target_sr=sr)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(sum(s.duration for s in self.samples), 2)
        return d


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")[:32]
    return f"{s or 'voix'}-{uuid.uuid4().hex[:6]}"


def prepare_sample(x: np.ndarray, sr: int) -> np.ndarray:
    """Nettoyage d'un échantillon : mono 24 kHz, passe-haut, silences coupés, normalisé."""
    x = audio.resample(audio.to_mono(x), sr, STORE_SR)
    x = audio.highpass(x, STORE_SR)
    x = audio.trim_silence(x, STORE_SR)
    x = audio.compress_pauses(x, STORE_SR)
    return audio.normalize_peak(x, 0.9)


class VoiceStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.VOICES_DIR
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ lecture
    def list(self) -> list[Voice]:
        voices = []
        if not self.root.exists():
            return voices
        for meta in sorted(self.root.glob("*/meta.json")):
            try:
                voices.append(self._load(meta))
            except Exception:
                continue
        return sorted(voices, key=lambda v: v.created_at, reverse=True)

    def get(self, voice_id: str) -> Voice:
        meta = self.root / voice_id / "meta.json"
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", voice_id) or not meta.exists():
            raise KeyError(f"Voix introuvable : {voice_id}")
        return self._load(meta)

    @staticmethod
    def _load(meta: Path) -> Voice:
        d = json.loads(meta.read_text(encoding="utf-8"))
        d["samples"] = [Sample(**s) for s in d.get("samples", [])]
        return Voice(**d)

    def save(self, voice: Voice) -> None:
        voice.updated_at = time.time()
        voice.dir.mkdir(parents=True, exist_ok=True)
        tmp = voice.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(asdict(voice), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(voice.dir / "meta.json")

    # ---------------------------------------------------------------- écriture
    def create(self, name: str, language: str = "fr", description: str = "",
               consent: dict | None = None) -> Voice:
        name = name.strip() or "Nouvelle voix"
        with self._lock:
            voice = Voice(id=_slug(name), name=name, language=language, description=description,
                          consent=consent or {})
            (voice.dir / "samples").mkdir(parents=True, exist_ok=True)
            self.save(voice)
            return voice

    def add_sample(self, voice_id: str, data: bytes, source: str = "upload",
                   original_name: str = "", transcript: str = "") -> Voice:
        x, sr = audio.load_audio(data)
        x = prepare_sample(x, sr)
        if len(x) < STORE_SR * 0.5:
            raise audio.AudioError("Échantillon trop court (moins d'une demi-seconde de voix).")
        with self._lock:
            voice = self.get(voice_id)
            idx = 1 + max([int(Path(s.file).stem) for s in voice.samples if Path(s.file).stem.isdigit()] or [0])
            rel = f"samples/{idx:03d}.wav"
            audio.save_wav(voice.dir / rel, x, STORE_SR)
            voice.samples.append(Sample(file=rel, duration=round(len(x) / STORE_SR, 2), source=source,
                                        original_name=original_name, transcript=transcript.strip()))
            self._rebuild_reference(voice)
            self.save(voice)
            return voice

    def remove_sample(self, voice_id: str, file: str) -> Voice:
        with self._lock:
            voice = self.get(voice_id)
            voice.samples = [s for s in voice.samples if s.file != file]
            (voice.dir / file).unlink(missing_ok=True)
            self._rebuild_reference(voice)
            self.save(voice)
            return voice

    def update(self, voice_id: str, **fields) -> Voice:
        with self._lock:
            voice = self.get(voice_id)
            for k in ("name", "language", "description", "transcript"):
                if fields.get(k) is not None:
                    setattr(voice, k, str(fields[k]).strip())
            if isinstance(fields.get("settings"), dict):
                voice.settings = {**voice.settings, **fields["settings"]}
            if fields.get("transcript") is not None:
                # la transcription influence certains conditionnements
                self.invalidate_cache(voice)
            self.save(voice)
            return voice

    def reorder(self, voice_id: str, files: list[str]) -> Voice:
        """Change l'ordre des échantillons (le premier sert de référence principale, ex. Chatterbox)."""
        with self._lock:
            voice = self.get(voice_id)
            rank = {f: i for i, f in enumerate(files)}
            voice.samples.sort(key=lambda s: rank.get(s.file, len(rank)))  # tri stable : inconnus à la fin
            self._rebuild_reference(voice)
            self.save(voice)
            return voice

    def set_transcripts(self, voice_id: str, texts: dict[str, str]) -> Voice:
        """Associe une transcription à des échantillons ({fichier: texte})."""
        with self._lock:
            voice = self.get(voice_id)
            for s in voice.samples:
                if s.file in texts:
                    s.transcript = texts[s.file].strip()
            self._rebuild_reference(voice)
            self.save(voice)
            return voice

    def delete(self, voice_id: str) -> None:
        voice = self.get(voice_id)
        shutil.rmtree(voice.dir, ignore_errors=True)

    def mark_prepared(self, voice_id: str, model_id: str) -> None:
        with self._lock:
            voice = self.get(voice_id)
            voice.prepared[model_id] = time.time()
            self.save(voice)

    def invalidate_cache(self, voice: Voice) -> None:
        shutil.rmtree(voice.dir / "cache", ignore_errors=True)
        voice.prepared = {}

    def _rebuild_reference(self, voice: Voice) -> None:
        """Assemble les échantillons en une référence unique (bornée en durée)."""
        self.invalidate_cache(voice)
        if not voice.samples:
            voice.reference_path.unlink(missing_ok=True)
            voice.analysis, voice.transcript = {}, ""
            return
        max_len = int(config.MAX_REFERENCE_SECONDS * STORE_SR)
        chunks, total, texts = [], 0, []
        complete = True  # chaque échantillon utilisé entier et transcrit ?
        gap = np.zeros(int(0.25 * STORE_SR), dtype=np.float32)
        for s in voice.samples:
            if total >= max_len:
                complete = False
                break
            x, _ = audio.load_audio(voice.dir / s.file)
            if len(x) > max_len - total:
                x, complete = x[: max_len - total], False
            chunks += [x, gap]
            total += len(x) + len(gap)
            if s.transcript:
                texts.append(s.transcript)
            else:
                complete = False
        ref = np.concatenate(chunks[:-1])
        audio.save_wav(voice.reference_path, ref, STORE_SR)
        voice.analysis = audio.analyze(ref, STORE_SR)
        # Une transcription partielle ferait halluciner F5-TTS : on la vide dans ce cas
        voice.transcript = " ".join(texts) if complete else ""
