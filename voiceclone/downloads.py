"""Téléchargement des modèles depuis Hugging Face, en tâche de fond avec suivi de progression."""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .registry import ModelSpec, get_model

log = logging.getLogger(__name__)

COMPLETE_MARKER = ".voiceclone_complete.json"


def model_dir(model_id: str) -> Path:
    return config.MODELS_DIR / model_id


def is_downloaded(model_id: str) -> bool:
    return (model_dir(model_id) / COMPLETE_MARKER).exists()


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


@dataclass
class DownloadState:
    model_id: str
    status: str = "pending"  # pending | running | done | error | cancelled
    total_bytes: int = 0
    downloaded_bytes: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    current_repo: str = ""

    def to_dict(self) -> dict:
        progress = 0.0
        if self.status == "done":
            progress = 1.0
        elif self.total_bytes:
            progress = min(0.99, self.downloaded_bytes / self.total_bytes)
        elapsed = (self.finished_at or time.time()) - self.started_at
        speed = self.downloaded_bytes / elapsed if elapsed > 0 else 0
        return {
            "status": self.status,
            "progress": round(progress, 4),
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "speed_bps": int(speed),
            "error": self.error,
            "current_repo": self.current_repo,
        }


def _matches(filename: str, patterns: tuple[str, ...] | None) -> bool:
    return patterns is None or any(fnmatch.fnmatch(filename, p) for p in patterns)


def _estimate_total(spec: ModelSpec) -> int:
    """Taille totale attendue (octets) via l'API Hub ; 0 si inconnue."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=config.HF_TOKEN)
        total = 0
        for repo in spec.repos:
            info = api.model_info(repo.repo_id, revision=repo.revision, files_metadata=True)
            for s in info.siblings or []:
                if _matches(s.rfilename, repo.allow_patterns) and s.size:
                    total += int(s.size)
        return total
    except Exception as exc:  # pragma: no cover - dépend du réseau
        log.warning("Impossible d'estimer la taille de %s : %s", spec.id, exc)
        return 0


class DownloadManager:
    def __init__(self) -> None:
        self._states: dict[str, DownloadState] = {}
        self._lock = threading.Lock()
        self._cancel: set[str] = set()

    def state(self, model_id: str) -> DownloadState | None:
        return self._states.get(model_id)

    def start(self, model_id: str) -> DownloadState:
        spec = get_model(model_id)
        with self._lock:
            current = self._states.get(model_id)
            if current and current.status in ("pending", "running"):
                return current
            st = DownloadState(model_id=model_id)
            self._states[model_id] = st
            self._cancel.discard(model_id)
        threading.Thread(target=self._run, args=(spec, st), daemon=True, name=f"dl-{model_id}").start()
        return st

    def cancel(self, model_id: str) -> None:
        self._cancel.add(model_id)

    def _run(self, spec: ModelSpec, st: DownloadState) -> None:
        from huggingface_hub import snapshot_download

        target = model_dir(spec.id)
        target.mkdir(parents=True, exist_ok=True)
        st.status = "running"
        st.total_bytes = _estimate_total(spec)
        baseline = dir_size(target)
        stop_poll = threading.Event()

        def poll() -> None:
            while not stop_poll.wait(0.5):
                st.downloaded_bytes = max(0, dir_size(target) - baseline)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        try:
            for repo in spec.repos:
                if spec.id in self._cancel:
                    raise InterruptedError("Téléchargement annulé")
                st.current_repo = repo.repo_id
                local = target / repo.subdir if repo.subdir else target
                snapshot_download(
                    repo_id=repo.repo_id,
                    revision=repo.revision,
                    allow_patterns=list(repo.allow_patterns) if repo.allow_patterns else None,
                    local_dir=str(local),
                    token=config.HF_TOKEN,
                )
            (target / COMPLETE_MARKER).write_text(json.dumps({
                "model_id": spec.id,
                "repos": [r.repo_id for r in spec.repos],
                "downloaded_at": time.time(),
            }))
            st.status = "done"
        except InterruptedError as exc:
            st.status, st.error = "cancelled", str(exc)
        except Exception as exc:
            log.exception("Échec du téléchargement de %s", spec.id)
            st.status, st.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            stop_poll.set()
            st.downloaded_bytes = max(st.downloaded_bytes, dir_size(target) - baseline)
            if st.status == "done" and not st.total_bytes:
                st.total_bytes = st.downloaded_bytes
            st.finished_at = time.time()

    def delete(self, model_id: str) -> None:
        get_model(model_id)
        st = self._states.get(model_id)
        if st and st.status in ("pending", "running"):
            raise RuntimeError("Téléchargement en cours, annulez-le d'abord.")
        shutil.rmtree(model_dir(model_id), ignore_errors=True)
        self._states.pop(model_id, None)
