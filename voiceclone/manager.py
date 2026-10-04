"""Chargement / déchargement des moteurs et sérialisation des inférences."""

from __future__ import annotations

import importlib
import logging
import re
import threading
import time
from collections import Counter

from . import device as devmod
from . import settings
from .downloads import DownloadManager, is_downloaded, model_dir
from .engines.base import Engine, EngineError
from .registry import ModelSpec, all_models, get_model

log = logging.getLogger(__name__)


class ModelNotReady(EngineError):
    pass


def size_gb(spec: ModelSpec) -> float:
    """Taille approximative d'un modèle d'après son catalogue (« ≈ 1.9 Go », « ≈ 130 Mo »)."""
    m = re.search(r"([\d.,]+)\s*(Go|Mo|GB|MB)", spec.size_hint or "")
    if not m:
        return 1.0
    v = float(m.group(1).replace(",", "."))
    return v if m.group(2) in ("Go", "GB") else v / 1024


class EngineManager:
    def __init__(self, downloads: DownloadManager | None = None) -> None:
        self.downloads = downloads or DownloadManager()
        self._engines: dict[str, Engine] = {}
        self._loading: dict[str, str] = {}  # model_id -> "loading" | message d'erreur
        self._load_lock = threading.Lock()
        # Un seul calcul à la fois sur le GPU : évite les OOM et les conflits CUDA
        self.infer_lock = threading.RLock()
        self._last_used: dict[str, float] = {}
        self._memory_gb: dict[str, float] = {}  # mémoire GPU mesurée au chargement
        self._pins: Counter = Counter()  # modèles utilisés par un Live : jamais déchargés automatiquement

    # ------------------------------------------------------------------ état
    def status(self, spec: ModelSpec) -> dict:
        dl = self.downloads.state(spec.id)
        missing = spec.missing_packages()
        return {
            **spec.to_dict(),
            "downloaded": is_downloaded(spec.id),
            "download": dl.to_dict() if dl else None,
            "missing_packages": missing,
            "installed": not missing,
            "loaded": spec.id in self._engines,
            "memory_gb": self._memory_gb.get(spec.id),
            "last_used": self._last_used.get(spec.id),
            "pinned": self._pins[spec.id] > 0,
            "loading": self._loading.get(spec.id) == "loading",
            "load_error": None if self._loading.get(spec.id) in (None, "loading") else self._loading[spec.id],
        }

    def list_status(self) -> list[dict]:
        return [self.status(s) for s in all_models()]

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._engines

    # ------------------------------------------------------------ chargement
    def get(self, model_id: str, capability: str | None = None) -> Engine:
        spec = get_model(model_id)
        if capability and capability not in spec.capabilities:
            raise EngineError(f"{spec.name} ne supporte pas « {capability} ».")
        self._last_used[model_id] = time.time()
        engine = self._engines.get(model_id)
        if engine is not None:
            return engine
        with self._load_lock:
            engine = self._engines.get(model_id)
            if engine is not None:
                return engine
            return self._load(spec)

    def _load(self, spec: ModelSpec) -> Engine:
        missing = spec.missing_packages()
        if missing:
            raise ModelNotReady(f"Dépendances manquantes pour {spec.name} : {', '.join(missing)}. "
                                f"Installez-les avec : {spec.pip}")
        if not is_downloaded(spec.id):
            raise ModelNotReady(f"{spec.name} n'est pas téléchargé. Téléchargez-le depuis l'onglet Modèles.")
        module_name, cls_name = spec.engine.split(":")
        module = importlib.import_module(module_name if "." in module_name else f"voiceclone.engines.{module_name}")
        engine_cls = getattr(module, cls_name)
        self._make_room(spec)
        engine: Engine = self._instantiate(spec, engine_cls)
        self._loading[spec.id] = "loading"
        t0 = time.time()
        before = devmod.allocated_gb(engine.device)
        try:
            with self.infer_lock:
                engine.load()
        except Exception as exc:
            self._loading[spec.id] = f"{type(exc).__name__}: {exc}"
            log.exception("Échec du chargement de %s", spec.id)
            raise EngineError(f"Échec du chargement de {spec.name} : {exc}") from exc
        self._loading.pop(spec.id, None)
        after = devmod.allocated_gb(engine.device)
        if before is not None and after is not None:
            self._memory_gb[spec.id] = round(max(0.0, after - before), 2)
        self._engines[spec.id] = engine
        self._last_used[spec.id] = time.time()
        log.info("%s chargé en %.1f s sur %s", spec.name, time.time() - t0, engine.device)
        return engine

    def load_async(self, model_id: str) -> None:
        get_model(model_id)

        def run():
            try:
                self.get(model_id)
            except Exception as exc:
                self._loading[model_id] = str(exc)

        self._loading[model_id] = "loading"
        threading.Thread(target=run, daemon=True).start()

    def _instantiate(self, spec: ModelSpec, engine_cls) -> Engine:
        python = (settings.get("engine_python") or {}).get(spec.id)
        if python:  # isolation : le moteur tourne dans un autre processus / environnement Python
            from .worker import RemoteEngine

            return RemoteEngine(spec, model_dir(spec.id), devmod.get_device(), python=python)
        return engine_cls(spec, model_dir(spec.id), devmod.get_device())

    # ----------------------------------------------------- mémoire / LRU
    def pin(self, *model_ids: str | None) -> None:
        for mid in filter(None, model_ids):
            self._pins[mid] += 1

    def unpin(self, *model_ids: str | None) -> None:
        for mid in filter(None, model_ids):
            self._pins[mid] -= 1
            if self._pins[mid] <= 0:
                del self._pins[mid]

    def free_vram_gb(self) -> float | None:
        return devmod.free_vram_gb()

    def _make_room(self, spec: ModelSpec) -> list[str]:
        """Décharge les modèles les moins récemment utilisés si nécessaire (nombre max, mémoire GPU)."""
        unloaded: list[str] = []
        if not settings.get("auto_unload"):
            return unloaded
        limit = int(settings.get("max_loaded_models") or 0)

        def candidates():
            return sorted((m for m in self._engines if self._pins[m] <= 0 and m != spec.id),
                          key=lambda m: self._last_used.get(m, 0))

        while limit and len(self._engines) >= limit and candidates():
            unloaded.append(self._evict(candidates()[0], "nombre maximal de modèles chargés atteint"))
        need = size_gb(spec) * 1.2
        free = self.free_vram_gb()
        while free is not None and free < need and candidates():
            unloaded.append(self._evict(candidates()[0], f"mémoire GPU libre {free:.1f} Go < {need:.1f} Go"))
            free = self.free_vram_gb()
        return unloaded

    def _evict(self, model_id: str, why: str) -> str:
        log.info("Déchargement automatique de %s (%s)", model_id, why)
        engine = self._engines.pop(model_id, None)
        if engine is not None:
            with self.infer_lock:
                engine.unload()
            devmod.free_memory()
        self._memory_gb.pop(model_id, None)
        return model_id

    def loaded(self) -> list[dict]:
        return [{"id": m, "memory_gb": self._memory_gb.get(m), "last_used": self._last_used.get(m),
                 "pinned": self._pins[m] > 0} for m in self._engines]

    def unload(self, model_id: str) -> None:
        with self._load_lock:
            engine = self._engines.pop(model_id, None)
        self._memory_gb.pop(model_id, None)
        if engine is not None:
            with self.infer_lock:
                engine.unload()
            devmod.free_memory()

    def unload_all(self) -> None:
        for mid in list(self._engines):
            self.unload(mid)
