"""Chargement / déchargement des moteurs et sérialisation des inférences."""

from __future__ import annotations

import importlib
import logging
import threading
import time

from . import device as devmod
from .downloads import DownloadManager, is_downloaded, model_dir
from .engines.base import Engine, EngineError
from .registry import ModelSpec, all_models, get_model

log = logging.getLogger(__name__)


class ModelNotReady(EngineError):
    pass


class EngineManager:
    def __init__(self, downloads: DownloadManager | None = None) -> None:
        self.downloads = downloads or DownloadManager()
        self._engines: dict[str, Engine] = {}
        self._loading: dict[str, str] = {}  # model_id -> "loading" | message d'erreur
        self._load_lock = threading.Lock()
        # Un seul calcul à la fois sur le GPU : évite les OOM et les conflits CUDA
        self.infer_lock = threading.RLock()

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
        engine: Engine = engine_cls(spec, model_dir(spec.id), devmod.get_device())
        self._loading[spec.id] = "loading"
        t0 = time.time()
        try:
            with self.infer_lock:
                engine.load()
        except Exception as exc:
            self._loading[spec.id] = f"{type(exc).__name__}: {exc}"
            log.exception("Échec du chargement de %s", spec.id)
            raise EngineError(f"Échec du chargement de {spec.name} : {exc}") from exc
        self._loading.pop(spec.id, None)
        self._engines[spec.id] = engine
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

    def unload(self, model_id: str) -> None:
        with self._load_lock:
            engine = self._engines.pop(model_id, None)
        if engine is not None:
            with self.infer_lock:
                engine.unload()
            devmod.free_memory()

    def unload_all(self) -> None:
        for mid in list(self._engines):
            self.unload(mid)
