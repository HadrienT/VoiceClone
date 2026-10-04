"""Isolation des moteurs : chaque modèle peut tourner dans son propre processus Python.

Utile quand deux modèles exigent des versions incompatibles d'une bibliothèque (transformers,
torch…) : on installe chaque moteur dans son venv et on indique dans Diagnostic → Réglages
quel Python utiliser pour quel modèle. VoiceClone dialogue alors avec ce processus.

Le venv du moteur doit contenir les paquets du modèle + numpy, scipy et soundfile ; le code de
VoiceClone y est rendu importable via PYTHONPATH.

Protocole : connexion locale authentifiée (multiprocessing.connection), messages picklés.
    -> ("init", spec, model_dir, device)          <- ("ok", None)
    -> ("call", méthode, args, kwargs)            <- ("ok", résultat) | ("err", type, message)
    -> ("stream", méthode, args, kwargs)          <- ("chunk", x)… puis ("end", None)
    -> ("close",)
"""

from __future__ import annotations

import importlib
import logging
import os
import secrets
import subprocess
import sys
import threading
import traceback
from multiprocessing.connection import Client, Listener
from pathlib import Path

from . import config
from .engines.base import Engine, EngineError

log = logging.getLogger(__name__)

START_TIMEOUT = float(os.environ.get("VOICECLONE_WORKER_TIMEOUT", "120"))


class RemoteEngine(Engine):
    """Mandataire : mêmes méthodes qu'un moteur, exécutées dans un processus séparé."""

    def __init__(self, spec, model_dir: Path, device: str, python: str | None = None) -> None:
        super().__init__(spec, model_dir, device)
        self.python = python or sys.executable
        self._proc: subprocess.Popen | None = None
        self._conn = None
        self._lock = threading.Lock()

    # -------------------------------------------------------- cycle de vie
    def load(self) -> None:
        if not Path(self.python).exists() and self.python != sys.executable:
            raise EngineError(f"Python introuvable pour {self.spec.name} : {self.python}")
        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        env = {**os.environ, "VOICECLONE_WORKER_ADDR": f"{listener.address[0]}:{listener.address[1]}",
               "VOICECLONE_WORKER_KEY": key.hex(),
               "PYTHONPATH": os.pathsep.join(filter(None, [str(config.ROOT_DIR), os.environ.get("PYTHONPATH")]))}
        self._proc = subprocess.Popen([self.python, "-m", "voiceclone.worker"], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._pump_logs, daemon=True).start()
        # accept() bloque : on le fait dans un thread pour pouvoir abandonner si le processus meurt
        box: dict = {}
        t = threading.Thread(target=lambda: box.setdefault("conn", listener.accept()), daemon=True)
        t.start()
        t.join(START_TIMEOUT)
        listener.close()
        if "conn" not in box:
            self._kill()
            raise EngineError(f"Le processus de {self.spec.name} n'a pas démarré (voir Diagnostic → Journal).")
        self._conn = box["conn"]
        self._request("init", self.spec, str(self.model_dir), self.device)

    def unload(self) -> None:
        try:
            if self._conn is not None:
                self._conn.send(("close",))
        except Exception:
            pass
        self._kill()

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc, self._conn = None, None

    def _pump_logs(self) -> None:
        proc = self._proc
        for line in proc.stdout if proc and proc.stdout else []:
            log.info("[%s] %s", self.spec.id, line.rstrip())

    # ------------------------------------------------------------ appels
    def _request(self, *msg):
        with self._lock:
            if self._conn is None:
                raise EngineError(f"{self.spec.name} n'est pas chargé.")
            try:
                self._conn.send(msg)
                reply = self._conn.recv()
            except (EOFError, OSError) as exc:
                self._kill()
                raise EngineError(f"Le processus de {self.spec.name} s'est arrêté : {exc}") from exc
        return self._unwrap(reply)

    @staticmethod
    def _unwrap(reply):
        if reply[0] == "err":
            raise EngineError(f"{reply[1]}: {reply[2]}")
        return reply[1]

    def _call(self, method: str, *args, **kwargs):
        return self._request("call", method, args, kwargs)

    def prepare_voice(self, voice):
        return self._call("prepare_voice", voice)

    def tts(self, text, voice, language="fr", **params):
        return self._call("tts", text, voice, language, **params)

    def convert(self, wav, sr, voice, **params):
        return self._call("convert", wav, sr, voice, **params)

    def transcribe(self, wav, sr, language=None, **params):
        return self._call("transcribe", wav, sr, language, **params)

    def translate(self, text, src, tgt):
        return self._call("translate", text, src, tgt)

    def tts_stream(self, text, voice, language="fr", **params):
        with self._lock:
            if self._conn is None:
                raise EngineError(f"{self.spec.name} n'est pas chargé.")
            self._conn.send(("stream", "tts_stream", (text, voice, language), params))
            while True:
                reply = self._conn.recv()
                if reply[0] == "end":
                    return
                if reply[0] == "chunk":
                    yield reply[1]
                else:
                    self._unwrap(reply)


# ---------------------------------------------------------- côté processus
def _serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    host, port = os.environ["VOICECLONE_WORKER_ADDR"].rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ["VOICECLONE_WORKER_KEY"]))
    engine = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        kind = msg[0]
        if kind == "close":
            break
        try:
            if kind == "init":
                spec, model_dir, device = msg[1:]
                module_name, cls_name = spec.engine.split(":")
                module = importlib.import_module(module_name if "." in module_name
                                                 else f"voiceclone.engines.{module_name}")
                engine = getattr(module, cls_name)(spec, Path(model_dir), device)
                engine.load()
                conn.send(("ok", None))
            elif kind == "call":
                _, method, args, kwargs = msg
                conn.send(("ok", getattr(engine, method)(*args, **kwargs)))
            elif kind == "stream":
                _, method, args, kwargs = msg
                for chunk in getattr(engine, method)(*args, **kwargs):
                    conn.send(("chunk", chunk))
                conn.send(("end", None))
        except Exception as exc:
            traceback.print_exc()
            conn.send(("err", type(exc).__name__, str(exc)))
    if engine is not None:
        try:
            engine.unload()
        except Exception:
            pass


if __name__ == "__main__":
    _serve()
