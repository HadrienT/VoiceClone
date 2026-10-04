"""File de tâches longues (livres audio, textes longs…) : exécution une par une, progression, annulation."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_KEPT = 100


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    title: str
    fn: Callable[[Job], dict | None] = field(repr=False)
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        """Appelé par la tâche ; lève JobCancelled si l'utilisateur a annulé."""
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            self.message = message
        self.check()

    def check(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self) -> dict:
        eta = None
        if self.state == "running" and self.started_at and 0.02 < self.progress < 1:
            eta = round((time.time() - self.started_at) * (1 - self.progress) / self.progress, 1)
        return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
                "progress": round(self.progress, 3), "message": self.message, "result": self.result,
                "error": self.error, "created_at": self.created_at, "started_at": self.started_at,
                "finished_at": self.finished_at, "eta_s": eta}


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="voiceclone-jobs")
        self._worker.start()

    def submit(self, kind: str, title: str, fn: Callable[[Job], dict | None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title, fn=fn)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Tâche introuvable : {job_id}")
        return job

    def list(self) -> list[dict]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job._cancel.set()
        if job.state == "queued":
            job.state, job.finished_at = "cancelled", time.time()
        return job

    def remove(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.state in ("queued", "running"):
            self.cancel(job_id)
        else:
            with self._lock:
                self._jobs.pop(job_id, None)

    def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        """Attend la fin d'une tâche (tests, scripts)."""
        end = time.time() + timeout
        job = self.get(job_id)
        while job.state in ("queued", "running") and time.time() < end:
            time.sleep(0.02)
        return job

    def _prune(self) -> None:
        done = sorted((j for j in self._jobs.values() if j.state not in ("queued", "running")),
                      key=lambda j: j.created_at)
        for j in done[: max(0, len(self._jobs) - MAX_KEPT)]:
            self._jobs.pop(j.id, None)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job.state != "queued":
                continue
            job.state, job.started_at = "running", time.time()
            try:
                job.result = job.fn(job) or {}
                job.state, job.progress = "done", 1.0
            except JobCancelled:
                job.state, job.message = "cancelled", "Annulée"
            except Exception as exc:
                log.exception("Tâche %s (%s) en échec", job.id, job.title)
                job.state, job.error = "error", f"{exc}"
            job.finished_at = time.time()
