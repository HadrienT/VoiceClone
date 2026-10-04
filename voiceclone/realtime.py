"""Moteur temps réel : micro -> modèle -> périphérique de sortie (câble virtuel pour Discord).

Trois modes :
- "vc"        : conversion de voix par morceaux (OpenVoice, Chatterbox VC). Garde vos intonations.
- "asr_tts"   : transcription (Whisper) puis synthèse dans la voix clonée (XTTS...). Plus de latence,
                mais n'importe quel modèle TTS fonctionne et la voix est 100 % celle du clone.
- "passthrough" : renvoie le micro tel quel, pour tester le routage audio vers Discord.

Pour Discord : choisissez comme sortie un câble audio virtuel (VB-CABLE sous Windows, BlackHole sous
macOS, sink PulseAudio/PipeWire sous Linux) et, dans Discord, ce même câble comme périphérique d'entrée.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import audio
from .manager import EngineManager
from .voices import VoiceStore

log = logging.getLogger(__name__)

VIRTUAL_HINTS = ("cable", "vb-audio", "voicemeeter", "blackhole", "loopback", "virtual", "voiceclone", "null")


class RealtimeUnavailable(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RealtimeUnavailable(
            f"Audio temps réel indisponible ({exc}). Installez `sounddevice` et la bibliothèque PortAudio "
            "(Linux : sudo apt install libportaudio2)."
        ) from exc
    return sd


def list_devices() -> dict:
    sd = _sd()
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    default_in, default_out = sd.default.device if sd.default.device is not None else (None, None)
    inputs, outputs = [], []
    for i, d in enumerate(devices):
        api = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else ""
        item = {
            "id": i,
            "name": d["name"],
            "hostapi": api,
            "samplerate": int(d["default_samplerate"]),
            "virtual": any(h in d["name"].lower() for h in VIRTUAL_HINTS),
        }
        if d["max_input_channels"] > 0:
            inputs.append({**item, "default": i == default_in, "channels": d["max_input_channels"]})
        if d["max_output_channels"] > 0:
            outputs.append({**item, "default": i == default_out, "channels": d["max_output_channels"]})
    return {"inputs": inputs, "outputs": outputs}


class AudioFifo:
    """File audio thread-safe lue par le callback de sortie."""

    def __init__(self, max_seconds: float, sr: int) -> None:
        self.sr = sr
        self.max = int(max_seconds * sr)
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._size = 0
        self._lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return
        with self._lock:
            self._chunks.append(np.asarray(x, dtype=np.float32))
            self._size += len(x)
            while self._size > self.max and self._chunks:  # trop de retard : on jette le plus ancien
                self._size -= len(self._chunks.popleft())

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        with self._lock:
            while pos < n and self._chunks:
                c = self._chunks[0]
                take = min(n - pos, len(c))
                out[pos:pos + take] = c[:take]
                pos += take
                if take == len(c):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = c[take:]
                self._size -= take
        return out

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._size = 0

    @property
    def seconds(self) -> float:
        return self._size / self.sr


def sola_offset(tail: np.ndarray, seg: np.ndarray, search: int) -> int:
    """Décalage (0..search) de `seg` qui maximise la corrélation normalisée avec `tail`."""
    n = len(tail)
    if search <= 0 or len(seg) < n + search or n == 0:
        return 0
    window = seg[: n + search]
    corr = np.correlate(window, tail, mode="valid")  # longueur search + 1
    energy = np.sqrt(np.convolve(window * window, np.ones(n), mode="valid")) + 1e-8
    return int(np.argmax(corr / energy))


@dataclass
class RealtimeConfig:
    mode: str = "vc"  # vc | asr_tts | passthrough
    model_id: str | None = None  # modèle VC ou TTS
    voice_id: str | None = None
    asr_model_id: str | None = None
    source_voice_id: str | None = None  # (optionnel) votre propre voix, améliore OpenVoice
    language: str = "fr"
    input_device: int | None = None
    output_device: int | None = None
    monitor_device: int | None = None  # pour vous entendre (casque)
    chunk_ms: int = 700
    context_ms: int = 300
    crossfade_ms: int = 40
    silence_db: float = -45.0
    end_silence_ms: int = 600
    input_gain: float = 1.0
    output_gain: float = 1.0
    params: dict = field(default_factory=dict)
    say_model_id: str | None = None  # modèle TTS pour le texte tapé (« Dire dans Discord »)
    warmup: bool = True  # inférence à blanc au démarrage : évite la latence du tout premier morceau


class RealtimeSession:
    def __init__(self, manager: EngineManager, voices: VoiceStore) -> None:
        self.manager = manager
        self.voices = voices
        self.cfg: RealtimeConfig | None = None
        self.state = "idle"
        self.error: str | None = None
        self._stop = threading.Event()
        self._streams: list = []
        self._threads: list[threading.Thread] = []
        self._in_q: queue.Queue[np.ndarray] = queue.Queue()
        self._out: AudioFifo | None = None
        self._mon: AudioFifo | None = None
        self.in_db = -120.0
        self.out_db = -120.0
        self.process_ms = 0.0
        self.chunks = 0
        self.dropped = 0
        self.transcripts: collections.deque[dict] = collections.deque(maxlen=20)
        self.started_at: float | None = None

    # ------------------------------------------------------------------ API
    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "config": asdict(self.cfg) if self.cfg else None,
            "input_db": round(self.in_db, 1),
            "output_db": round(self.out_db, 1),
            "process_ms": round(self.process_ms, 1),
            "buffer_ms": round((self._out.seconds if self._out else 0) * 1000),
            "chunks": self.chunks,
            "dropped": self.dropped,
            "transcripts": list(self.transcripts),
            "latency_ms": self.estimated_latency_ms(),
            "uptime": round(time.time() - self.started_at, 1) if self.started_at and self.state == "running" else 0,
        }

    def estimated_latency_ms(self) -> int:
        """Latence côté serveur : attente d'un morceau + calcul + tampon de sortie (hors réseau)."""
        if not self.cfg or self.state != "running":
            return 0
        buffer_ms = (self._out.seconds if self._out else 0) * 1000
        if self.cfg.mode == "vc":
            return round(self.cfg.chunk_ms + self.process_ms + buffer_ms)
        if self.cfg.mode == "asr_tts":
            last = next((t for t in reversed(self.transcripts) if "first_audio_ms" in t), None)
            return round(self.cfg.end_silence_ms + (last["first_audio_ms"] if last else 0) + buffer_ms)
        return round(20 + buffer_ms)

    def say(self, text: str) -> None:
        """Fait dire un texte tapé à la voix clonée, mêlé au flux du Live (ex. vers Discord)."""
        text = (text or "").strip()
        if self.state != "running" or not text:
            raise ValueError("Démarrez le Live et tapez un texte.")
        model_id = self.cfg.say_model_id or (self.cfg.model_id if self.cfg.mode == "asr_tts" else None)
        if not model_id:
            raise ValueError("Choisissez le modèle de synthèse pour le texte tapé.")
        if not getattr(self, "voice", None):
            if not self.cfg.voice_id:
                raise ValueError("Choisissez une voix.")
            self.voice = self.voices.get(self.cfg.voice_id)
        entry = {"text": text, "at": time.time(), "typed": True}
        self.transcripts.append(entry)

        def run():
            engine = self.manager.get(model_id, "tts")
            t0 = time.perf_counter()
            gen = engine.tts_stream(text, self.voice, self.cfg.language, **self.cfg.params)
            while not self._stop.is_set():
                with self.manager.infer_lock:
                    piece = next(gen, None)
                if piece is None:
                    break
                entry.setdefault("first_audio_ms", round((time.perf_counter() - t0) * 1000))
                self._emit(*piece)

        t = threading.Thread(target=self._guard_say(run), daemon=True, name="rt-say")
        self._threads.append(t)
        t.start()

    def _guard_say(self, fn):
        def run():
            try:
                fn()
            except Exception as exc:  # une phrase ratée ne doit pas couper le Live
                log.exception("Échec du texte tapé")
                self.transcripts.append({"text": f"⚠ {exc}", "at": time.time(), "typed": True})
        return run

    def start(self, cfg: RealtimeConfig) -> None:
        if self.state in ("starting", "running"):
            self.stop()
        self._check_io()
        self.cfg = cfg
        self.state, self.error = "starting", None
        self._stop.clear()
        self._in_q = queue.Queue()
        self.chunks = self.dropped = 0
        self.transcripts.clear()
        try:
            self._prepare_models(cfg)
            self._open_io(cfg)
        except Exception as exc:
            self._close_streams()
            self.state, self.error = "error", str(exc)
            raise
        worker = {"vc": self._vc_loop, "asr_tts": self._asr_loop, "passthrough": self._passthrough_loop}[cfg.mode]
        t = threading.Thread(target=self._guard(worker), daemon=True, name="rt-worker")
        self._threads = [t]
        t.start()
        self.started_at = time.time()
        self.state = "running"

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        self._close_streams()
        if self.state != "error":
            self.state = "idle"
        self.in_db = self.out_db = -120.0

    # ------------------------------------------------------------- internes
    def _guard(self, fn):
        def run():
            try:
                fn()
            except Exception as exc:  # pragma: no cover - dépend du matériel
                log.exception("Erreur dans la boucle temps réel")
                self.error, self.state = str(exc), "error"
                self._close_streams()
        return run

    def _prepare_models(self, cfg: RealtimeConfig) -> None:
        if cfg.mode not in ("vc", "asr_tts", "passthrough"):
            raise ValueError(f"Mode inconnu : {cfg.mode}")
        if cfg.mode == "passthrough":
            return
        if not cfg.voice_id or not cfg.model_id:
            raise ValueError("Choisissez un modèle et une voix.")
        self.voice = self.voices.get(cfg.voice_id)
        self.source_voice = self.voices.get(cfg.source_voice_id) if cfg.source_voice_id else None
        if cfg.mode == "vc":
            self.engine = self.manager.get(cfg.model_id, "vc")
        else:
            if not cfg.asr_model_id:
                raise ValueError("Choisissez un modèle de transcription (Whisper).")
            self.asr = self.manager.get(cfg.asr_model_id, "asr")
            self.engine = self.manager.get(cfg.model_id, "tts")
        with self.manager.infer_lock:
            self.engine.prepare_voice(self.voice)
            if cfg.warmup:
                self._warmup(cfg)
        if hasattr(self.engine, "reset_stream"):
            self.engine.reset_stream()

    def _warmup(self, cfg: RealtimeConfig) -> None:
        """Inférence à blanc : charge les noyaux GPU / caches pour que le 1er vrai morceau soit rapide."""
        rng = np.random.default_rng(0)
        x = (0.01 * rng.standard_normal(16000 // 2)).astype(np.float32)
        try:
            if cfg.mode == "vc":
                self.engine.convert(x, 16000, self.voice, **cfg.params)
            else:
                self.asr.transcribe(x, 16000, cfg.language)
        except Exception as exc:  # le préchauffage est une optimisation : il ne doit jamais bloquer
            log.warning("Préchauffage ignoré : %s", exc)

    def _check_io(self) -> None:
        """Vérifie au plus tôt que l'audio est disponible (avant de charger les modèles)."""
        _sd()

    def _open_io(self, cfg: RealtimeConfig) -> None:
        self._open_streams(_sd(), cfg)

    def _open_streams(self, sd, cfg: RealtimeConfig) -> None:
        in_info = sd.query_devices(cfg.input_device, "input")
        out_info = sd.query_devices(cfg.output_device, "output")
        self.in_sr = int(in_info["default_samplerate"])
        self.out_sr = int(out_info["default_samplerate"])
        self._out = AudioFifo(4.0, self.out_sr)
        block = int(self.in_sr * 0.02)

        def on_input(indata, frames, t, status):
            x = indata[:, 0].astype(np.float32) * cfg.input_gain
            self.in_db = 0.8 * self.in_db + 0.2 * audio.rms_db(x)
            self._in_q.put(x.copy())

        def make_output(fifo: AudioFifo, track_level: bool):
            def on_output(outdata, frames, t, status):
                x = fifo.read(frames) * cfg.output_gain
                if track_level:
                    self.out_db = 0.8 * self.out_db + 0.2 * audio.rms_db(x)
                outdata[:] = np.clip(x, -1, 1)[:, None]
            return on_output

        out_ch = min(2, int(out_info["max_output_channels"])) or 1
        streams = [
            sd.InputStream(device=cfg.input_device, channels=1, samplerate=self.in_sr, blocksize=block,
                           dtype="float32", callback=on_input, latency="low"),
            sd.OutputStream(device=cfg.output_device, channels=out_ch, samplerate=self.out_sr,
                            dtype="float32", callback=make_output(self._out, True), latency="low"),
        ]
        self._mon = None
        if cfg.monitor_device is not None and cfg.monitor_device != cfg.output_device:
            mon_info = sd.query_devices(cfg.monitor_device, "output")
            self.mon_sr = int(mon_info["default_samplerate"])
            self._mon = AudioFifo(4.0, self.mon_sr)
            streams.append(sd.OutputStream(
                device=cfg.monitor_device, channels=min(2, int(mon_info["max_output_channels"])) or 1,
                samplerate=self.mon_sr, dtype="float32", callback=make_output(self._mon, False), latency="low"))
        self._streams = streams
        for s in streams:
            s.start()

    def _close_streams(self) -> None:
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []

    def _emit(self, x: np.ndarray, sr: int) -> None:
        if len(x) == 0:
            return
        self._out.push(audio.resample(x, sr, self.out_sr))
        if self._mon is not None:
            self._mon.push(audio.resample(x, sr, self.mon_sr))

    def _read_input(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._in_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------------------------------------------------------- boucles
    def _passthrough_loop(self) -> None:
        while not self._stop.is_set():
            x = self._read_input()
            if x is not None:
                self._emit(x, self.in_sr)

    def _vc_loop(self) -> None:
        cfg = self.cfg
        sr = self.in_sr
        chunk_n = int(sr * cfg.chunk_ms / 1000)
        ctx_n = int(sr * cfg.context_ms / 1000)
        params = {**cfg.params, "stream": True, "source_voice": self.source_voice}
        pending = np.zeros(0, dtype=np.float32)
        context = np.zeros(0, dtype=np.float32)
        prev_tail: np.ndarray | None = None
        out_sr = None
        hangover = 0  # nombre de morceaux encore traités après la fin de la parole

        while not self._stop.is_set():
            x = self._read_input()
            if x is None:
                continue
            blocks = [pending, x]
            while not self._in_q.empty():  # récupère tout le retard accumulé pendant l'inférence
                blocks.append(self._in_q.get_nowait())
            pending = np.concatenate(blocks)
            # Si le traitement prend du retard, on saute de l'audio pour rester en direct
            if len(pending) > 3 * chunk_n:
                skip = len(pending) - chunk_n
                pending = pending[skip:]
                self.dropped += 1
            if len(pending) < chunk_n:
                continue
            chunk, pending = pending[:chunk_n], pending[chunk_n:]

            voiced = audio.rms_db(chunk) > cfg.silence_db
            if voiced:
                hangover = 1
            elif hangover > 0:
                hangover -= 1  # on traite encore ce morceau pour ne pas couper la fin des mots
            else:
                if prev_tail is not None and out_sr:
                    self._emit(prev_tail * np.linspace(1, 0, len(prev_tail), dtype=np.float32), out_sr)
                    prev_tail = None
                context = chunk[-ctx_n:] if ctx_n else context[:0]
                continue

            seg_in = np.concatenate([context, chunk])
            t0 = time.perf_counter()
            with self.manager.infer_lock:
                y, out_sr = self.engine.convert(seg_in, sr, self.voice, **params)
            self.process_ms = (time.perf_counter() - t0) * 1000
            self.chunks += 1

            # On ne garde que la partie de sortie correspondant au nouveau morceau (+ fondu)
            fade_n = int(out_sr * cfg.crossfade_ms / 1000)
            ratio = len(y) / max(1, len(seg_in))
            start = max(0, int(len(context) * ratio) - fade_n)
            context = chunk[-ctx_n:] if ctx_n else context[:0]
            seg = y[start:].astype(np.float32)
            if len(seg) <= 2 * fade_n + int(out_sr * 0.012):
                continue
            if prev_tail is not None and len(prev_tail) == fade_n and fade_n:
                # Alignement (SOLA) : décale le début du morceau pour qu'il soit en phase avec la fin
                # du précédent, ce qui évite l'effet « robot » / les annulations dans le fondu
                k = sola_offset(prev_tail, seg, int(out_sr * 0.012))
                seg = seg[k:]
                ramp = np.linspace(0, 1, fade_n, dtype=np.float32)
                seg[:fade_n] = prev_tail * (1 - ramp) + seg[:fade_n] * ramp
            elif fade_n:
                seg[:fade_n] *= np.linspace(0, 1, fade_n, dtype=np.float32)
            if fade_n:
                prev_tail = seg[-fade_n:].copy()
                seg = seg[:-fade_n]
            self._emit(seg, out_sr)

    def _asr_loop(self) -> None:
        """Détection de phrases (VAD par énergie) -> Whisper -> TTS en streaming."""
        cfg = self.cfg
        sr = self.in_sr
        utterances: queue.Queue[np.ndarray] = queue.Queue()
        synth = threading.Thread(target=self._guard(lambda: self._synth_loop(utterances)), daemon=True)
        synth.start()
        self._threads.append(synth)

        preroll: collections.deque[np.ndarray] = collections.deque(maxlen=10)  # ~200 ms
        speech: list[np.ndarray] = []
        silence_ms = 0.0
        speech_ms = 0.0
        while not self._stop.is_set():
            x = self._read_input()
            if x is None:
                continue
            dur = len(x) / sr * 1000
            voiced = audio.rms_db(x) > cfg.silence_db
            if not speech:
                preroll.append(x)
                if voiced:
                    speech = list(preroll)
                    speech_ms, silence_ms = dur, 0.0
                continue
            speech.append(x)
            speech_ms += dur
            silence_ms = 0.0 if voiced else silence_ms + dur
            if silence_ms >= cfg.end_silence_ms or speech_ms > 15000:
                if speech_ms - silence_ms >= 300:
                    utterances.put(np.concatenate(speech))
                speech, silence_ms, speech_ms = [], 0.0, 0.0
                preroll.clear()

    def _synth_loop(self, utterances: queue.Queue) -> None:
        cfg = self.cfg
        while not self._stop.is_set():
            try:
                utt = utterances.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            with self.manager.infer_lock:
                text = self.asr.transcribe(utt, self.in_sr, cfg.language)
            if not text:
                continue
            entry = {"text": text, "at": time.time(), "asr_ms": round((time.perf_counter() - t0) * 1000)}
            self.transcripts.append(entry)
            first = True
            gen = self.engine.tts_stream(text, self.voice, cfg.language, **cfg.params)
            while True:
                with self.manager.infer_lock:
                    piece = next(gen, None)
                if piece is None or self._stop.is_set():
                    break
                wav, wsr = piece
                if first:
                    entry["first_audio_ms"] = round((time.perf_counter() - t0) * 1000)
                    first = False
                self._emit(wav, wsr)
            self.process_ms = (time.perf_counter() - t0) * 1000
            self.chunks += 1


class BrowserRealtimeSession(RealtimeSession):
    """Session live dont l'audio passe par le navigateur (WebSocket) au lieu des périphériques du serveur.

    Le navigateur capte le micro du PC de l'utilisateur, envoie le PCM via `feed()`, et rejoue ce que
    `on_audio` reçoit (PCM 16 bits mono à OUT_SR Hz) sur la sortie de son choix (ex. CABLE Input
    pour Discord). Indispensable quand VoiceClone tourne sur un serveur distant.
    """

    OUT_SR = 24000

    def __init__(self, manager: EngineManager, voices: VoiceStore, in_sr: int, on_audio,
                 out_sr: int | None = None) -> None:
        super().__init__(manager, voices)
        self.in_sr = int(in_sr)
        self.out_sr = int(out_sr or self.OUT_SR)
        self.on_audio = on_audio  # reçoit du float32 mono à out_sr

    def _check_io(self) -> None:
        pass  # aucun périphérique côté serveur

    def _open_io(self, cfg: RealtimeConfig) -> None:
        self._out = self._mon = None

    def feed(self, x: np.ndarray) -> None:
        """Audio du micro reçu du navigateur (float32 mono à in_sr)."""
        if self.state != "running":
            return
        x = np.asarray(x, dtype=np.float32) * self.cfg.input_gain
        self.in_db = 0.8 * self.in_db + 0.2 * audio.rms_db(x)
        self._in_q.put(x)

    def _emit(self, x: np.ndarray, sr: int) -> None:
        if len(x) == 0:
            return
        y = np.clip(audio.resample(x, sr, self.out_sr) * self.cfg.output_gain, -1.0, 1.0).astype(np.float32)
        self.out_db = 0.8 * self.out_db + 0.2 * audio.rms_db(y)
        self.on_audio(y)
