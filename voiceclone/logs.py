"""Journal du serveur conservé en mémoire pour la page Diagnostic."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

_MAX = 2000


class RingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: deque[dict] = deque(maxlen=_MAX)
        self.counter = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += "\n" + logging.Formatter().formatException(record.exc_info)
        except Exception:
            msg = str(record.msg)
        with self._lock:
            self.counter += 1
            self.records.append({"n": self.counter, "t": record.created or time.time(),
                                 "level": record.levelname, "name": record.name, "msg": msg[:4000]})

    def since(self, after: int = 0, min_level: str = "INFO") -> list[dict]:
        lvl = logging.getLevelName(min_level.upper())
        lvl = lvl if isinstance(lvl, int) else logging.INFO
        with self._lock:
            return [r for r in self.records if r["n"] > after and logging.getLevelName(r["level"]) >= lvl]


HANDLER = RingHandler()


def install() -> RingHandler:
    """Branche le tampon sur le journal racine (une seule fois)."""
    root = logging.getLogger()
    if HANDLER not in root.handlers:
        root.addHandler(HANDLER)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
    for name in ("uvicorn.error",):  # uvicorn n'envoie pas toujours vers la racine
        lg = logging.getLogger(name)
        if HANDLER not in lg.handlers and not lg.propagate:
            lg.addHandler(HANDLER)
    return HANDLER
