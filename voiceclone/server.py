"""Application FastAPI : API REST + interface web."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, audio, config
from . import device as devmod
from .downloads import DownloadManager
from .engines.base import EngineError
from .history import History
from .manager import EngineManager, ModelNotReady
from .realtime import (
    BrowserRealtimeSession,
    RealtimeConfig,
    RealtimeSession,
    RealtimeUnavailable,
    list_devices,
)
from .registry import get_model
from .voices import VoiceStore

log = logging.getLogger("voiceclone")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


# ---------------------------------------------------------------- schémas
class TTSRequest(BaseModel):
    model_id: str
    voice_id: str
    text: str = Field(min_length=1, max_length=5000)
    language: str = "fr"
    params: dict = Field(default_factory=dict)


class VoiceUpdate(BaseModel):
    name: str | None = None
    language: str | None = None
    description: str | None = None
    transcript: str | None = None


class TranscriptsUpdate(BaseModel):
    texts: dict[str, str]  # nom du fichier d'échantillon (ex. "001.wav") -> texte prononcé


class PrepareRequest(BaseModel):
    model_id: str


class TranscribeRequest(BaseModel):
    model_id: str


class RealtimeStart(BaseModel):
    mode: str = "vc"
    model_id: str | None = None
    voice_id: str | None = None
    asr_model_id: str | None = None
    source_voice_id: str | None = None
    language: str = "fr"
    input_device: int | None = None
    output_device: int | None = None
    monitor_device: int | None = None
    chunk_ms: int = Field(700, ge=200, le=3000)
    context_ms: int = Field(300, ge=0, le=2000)
    crossfade_ms: int = Field(40, ge=0, le=200)
    silence_db: float = Field(-45.0, ge=-90, le=0)
    end_silence_ms: int = Field(600, ge=200, le=3000)
    input_gain: float = Field(1.0, ge=0, le=10)
    output_gain: float = Field(1.0, ge=0, le=10)
    params: dict = Field(default_factory=dict)


# ------------------------------------------------------------------- app
def create_app() -> FastAPI:
    config.ensure_dirs()
    downloads = DownloadManager()
    manager = EngineManager(downloads)
    voices = VoiceStore()
    history = History()
    live = RealtimeSession(manager, voices)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        live.stop()
        manager.unload_all()

    app = FastAPI(title="VoiceClone", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def _no_stale_ui(request: Request, call_next):
        # Sans cette consigne, le navigateur peut garder l'ancien app.js après une mise à jour
        # (git pull) : il revalide désormais chaque fichier de l'interface (réponse 304 si inchangé).
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response
    app.state.manager = manager
    app.state.voices = voices
    app.state.history = history
    app.state.live = live

    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)

    @app.exception_handler(ModelNotReady)
    async def _not_ready(_: Request, exc: ModelNotReady):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(EngineError)
    async def _engine(_: Request, exc: EngineError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(audio.AudioError)
    async def _audio(_: Request, exc: audio.AudioError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        # Message réel renvoyé à l'interface (au lieu d'un « 500 Internal Server Error » opaque)
        log.exception("Erreur inattendue")
        return JSONResponse({"detail": f"Erreur interne ({type(exc).__name__}) : {exc}"}, status_code=500)

    @app.exception_handler(RealtimeUnavailable)
    async def _rt(_: Request, exc: RealtimeUnavailable):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    async def read_upload(f: UploadFile) -> bytes:
        data = await f.read()
        if not data:
            raise HTTPException(400, "Fichier vide.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Fichier trop volumineux (200 Mo max).")
        return data

    # ------------------------------------------------------------ système
    @app.get("/api/system")
    def system():
        return {"version": __version__, "data_dir": str(config.DATA_DIR), **devmod.system_info()}

    # ------------------------------------------------------------- modèles
    @app.get("/api/models")
    def models():
        return manager.list_status()

    @app.post("/api/models/{model_id}/download")
    def download(model_id: str):
        manager.downloads.start(model_id)
        return manager.status(get_model(model_id))

    @app.post("/api/models/{model_id}/cancel")
    def cancel(model_id: str):
        manager.downloads.cancel(model_id)
        return manager.status(get_model(model_id))

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str):
        manager.unload(model_id)
        try:
            manager.downloads.delete(model_id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return manager.status(get_model(model_id))

    @app.post("/api/models/{model_id}/load")
    def load(model_id: str):
        manager.load_async(model_id)
        return manager.status(get_model(model_id))

    @app.post("/api/models/{model_id}/unload")
    def unload(model_id: str):
        manager.unload(model_id)
        return manager.status(get_model(model_id))

    # --------------------------------------------------------------- voix
    @app.get("/api/voices")
    def list_voices():
        return [v.to_dict() for v in voices.list()]

    @app.post("/api/voices")
    async def create_voice(
        name: str = Form(...),
        language: str = Form("fr"),
        description: str = Form(""),
        consent: bool = Form(False),
        transcript: str = Form(""),
        source: str = Form("upload"),
        files: list[UploadFile] = File(default=[]),
    ):
        if not consent:
            raise HTTPException(400, "Vous devez confirmer avoir le droit d'utiliser cette voix.")
        payloads = [(await read_upload(f), f.filename or "") for f in files]
        voice = voices.create(name, language, description)
        try:
            for data, fname in payloads:
                voice = voices.add_sample(voice.id, data, source=source, original_name=fname,
                                          transcript=transcript if len(payloads) == 1 else "")
        except Exception:
            voices.delete(voice.id)
            raise
        return voice.to_dict()

    @app.get("/api/voices/{voice_id}")
    def get_voice(voice_id: str):
        return voices.get(voice_id).to_dict()

    @app.patch("/api/voices/{voice_id}")
    def update_voice(voice_id: str, body: VoiceUpdate):
        return voices.update(voice_id, **body.model_dump()).to_dict()

    @app.delete("/api/voices/{voice_id}")
    def delete_voice(voice_id: str):
        voices.delete(voice_id)
        return {"ok": True}

    @app.post("/api/voices/{voice_id}/samples")
    async def add_sample(voice_id: str, file: UploadFile = File(...), source: str = Form("upload"),
                         transcript: str = Form("")):
        data = await read_upload(file)
        return voices.add_sample(voice_id, data, source=source, original_name=file.filename or "",
                                 transcript=transcript).to_dict()

    @app.post("/api/voices/{voice_id}/auto-import")
    async def auto_import(
        voice_id: str,
        file: UploadFile = File(...),
        source: str = Form("upload"),
        enhance: bool = Form(True),
        target_seconds: float = Form(30.0),
        replace: bool = Form(False),
        transcribe_model_id: str = Form(""),
    ):
        """Import intelligent : nettoie l'enregistrement, le découpe, ne garde que les meilleurs passages
        (le meilleur en premier) et, si un modèle Whisper est indiqué, les transcrit."""
        from starlette.concurrency import run_in_threadpool

        from .prep import auto_prepare

        data = await read_upload(file)
        name = Path(file.filename or "enregistrement").stem

        def work():
            voices.get(voice_id)
            x, sr = audio.load_audio(data)
            pieces, report = auto_prepare(x, sr, enhance=enhance, target_s=max(5.0, min(target_seconds, 120.0)))
            if replace:
                for smp in list(voices.get(voice_id).samples):
                    voices.remove_sample(voice_id, smp.file)
            texts = {}
            asr = manager.get(transcribe_model_id, "asr") if transcribe_model_id else None
            for rank, piece in enumerate(pieces, 1):
                voice = voices.add_sample(voice_id, audio.to_wav_bytes(piece, sr), source=source,
                                          original_name=f"{name} · meilleur n°{rank}")
                if asr is not None:
                    with manager.infer_lock:
                        texts[voice.samples[-1].file] = asr.transcribe(piece, sr, voice.language)
            voice = voices.set_transcripts(voice_id, texts) if texts else voices.get(voice_id)
            return voice, report

        voice, report = await run_in_threadpool(work)
        return {"voice": voice.to_dict(), "report": report}

    @app.delete("/api/voices/{voice_id}/samples/{name}")
    def delete_sample(voice_id: str, name: str):
        return voices.remove_sample(voice_id, f"samples/{name}").to_dict()

    @app.put("/api/voices/{voice_id}/transcripts")
    def update_transcripts(voice_id: str, body: TranscriptsUpdate):
        """Texte prononcé dans chaque échantillon (saisi à la main ou corrigé après Whisper)."""
        return voices.set_transcripts(voice_id, {f"samples/{k}": v for k, v in body.texts.items()}).to_dict()

    @app.get("/api/voices/{voice_id}/audio")
    def voice_audio(voice_id: str, sample: str | None = None):
        voice = voices.get(voice_id)
        if sample:
            match = [s for s in voice.samples if s.file == f"samples/{sample}"]
            if not match:
                raise KeyError("Échantillon introuvable")
            path = voice.dir / match[0].file
        else:
            path = voice.reference_path
        if not path.exists():
            raise KeyError("Aucun audio pour cette voix")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/voices/{voice_id}/prepare")
    def prepare_voice(voice_id: str, body: PrepareRequest):
        """« Entraînement » : pré-calcule l'empreinte de la voix pour un modèle."""
        voice = voices.get(voice_id)
        engine = manager.get(body.model_id)
        t0 = time.time()
        with manager.infer_lock:
            engine.prepare_voice(voice)
        voices.mark_prepared(voice_id, body.model_id)
        return {"ok": True, "seconds": round(time.time() - t0, 2), "voice": voices.get(voice_id).to_dict()}

    @app.post("/api/voices/{voice_id}/transcribe")
    def transcribe_voice(voice_id: str, body: TranscribeRequest):
        """Transcrit automatiquement les échantillons (utile pour F5-TTS)."""
        voice = voices.get(voice_id)
        engine = manager.get(body.model_id, "asr")
        texts = {}
        for s in voice.samples:
            if s.transcript:
                continue
            x, sr = audio.load_audio(voice.dir / s.file)
            with manager.infer_lock:
                texts[s.file] = engine.transcribe(x, sr, voice.language)
        return voices.set_transcripts(voice_id, texts).to_dict()

    # ---------------------------------------------------------------- TTS
    @app.post("/api/tts")
    def tts(req: TTSRequest):
        voice = voices.get(req.voice_id)
        engine = manager.get(req.model_id, "tts")
        t0 = time.time()
        with manager.infer_lock:
            wav, sr = engine.tts(req.text, voice, req.language, **req.params)
        elapsed = time.time() - t0
        item = history.add(wav, sr, kind="tts", model_id=req.model_id, voice_id=voice.id, voice_name=voice.name,
                           text=req.text[:500], language=req.language, seconds=round(elapsed, 2))
        return Response(audio.to_wav_bytes(wav, sr), media_type="audio/wav", headers={
            "X-History-Id": item["id"], "X-Generation-Seconds": f"{elapsed:.2f}",
            "X-Audio-Seconds": f"{len(wav) / sr:.2f}",
        })

    # ----------------------------------------------------- speech-to-speech
    @app.post("/api/vc")
    async def vc(
        file: UploadFile = File(...),
        model_id: str = Form(...),
        voice_id: str = Form(...),
        mode: str = Form("vc"),  # vc | asr_tts
        asr_model_id: str = Form(""),
        language: str = Form("fr"),
        source_voice_id: str = Form(""),
        params: str = Form("{}"),
    ):
        import json

        from starlette.concurrency import run_in_threadpool

        data = await read_upload(file)
        extra = json.loads(params or "{}")

        def work():
            voice = voices.get(voice_id)
            x, sr = audio.load_audio(data)
            t0 = time.time()
            text = None
            if mode == "asr_tts":
                if not asr_model_id:
                    raise HTTPException(400, "Choisissez un modèle de transcription.")
                asr = manager.get(asr_model_id, "asr")
                tts_engine = manager.get(model_id, "tts")
                with manager.infer_lock:
                    text = asr.transcribe(x, sr, language)
                if not text:
                    raise HTTPException(400, "Aucune parole détectée dans l'audio.")
                with manager.infer_lock:
                    wav, out_sr = tts_engine.tts(text, voice, language, **extra)
            else:
                engine = manager.get(model_id, "vc")
                if source_voice_id:
                    extra["source_voice"] = voices.get(source_voice_id)
                with manager.infer_lock:
                    wav, out_sr = engine.convert(x, sr, voice, **extra)
            return wav, out_sr, text, time.time() - t0, voice

        wav, sr, text, elapsed, voice = await run_in_threadpool(work)
        item = history.add(wav, sr, kind="s2s", mode=mode, model_id=model_id, voice_id=voice.id,
                           voice_name=voice.name, text=(text or "")[:500], seconds=round(elapsed, 2))
        headers = {"X-History-Id": item["id"], "X-Generation-Seconds": f"{elapsed:.2f}"}
        if text:
            headers["X-Transcript"] = quote(text[:500])
        return Response(audio.to_wav_bytes(wav, sr), media_type="audio/wav", headers=headers)

    # ------------------------------------------------------------ historique
    @app.get("/api/history")
    def list_history(limit: int = 50):
        return history.list(limit)

    @app.get("/api/history/{item_id}/audio")
    def history_audio(item_id: str):
        return FileResponse(history.path(item_id), media_type="audio/wav", filename=f"voiceclone-{item_id}.wav")

    @app.delete("/api/history/{item_id}")
    def delete_history(item_id: str):
        history.delete(item_id)
        return {"ok": True}

    # ------------------------------------------------------------ temps réel
    @app.get("/api/realtime/devices")
    def devices():
        return list_devices()

    @app.post("/api/realtime/start")
    def rt_start(body: RealtimeStart):
        try:
            live.start(RealtimeConfig(**body.model_dump()))
        except (RealtimeUnavailable, KeyError, EngineError):
            raise
        except Exception as exc:
            raise HTTPException(400, f"Impossible de démarrer : {exc}") from exc
        return live.status()

    @app.post("/api/realtime/stop")
    def rt_stop():
        live.stop()
        return live.status()

    @app.get("/api/realtime/status")
    def rt_status():
        return live.status()

    @app.websocket("/api/realtime/ws")
    async def rt_browser(ws: WebSocket):
        """Live via le navigateur : micro du PC -> serveur (GPU) -> sortie choisie dans le navigateur.

        Protocole : 1er message texte = configuration JSON (+ "sample_rate" du micro), puis des
        messages binaires PCM int16 mono. Le serveur renvoie du PCM int16 mono à 24 kHz et des
        messages JSON {"type": "status" | "started" | "error", ...}.
        """
        from starlette.concurrency import run_in_threadpool

        await ws.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue = asyncio.Queue()  # bytes (audio) ou dict (JSON), envoyés dans l'ordre
        session: BrowserRealtimeSession | None = None

        async def sender():
            while True:
                item = await outbox.get()
                if isinstance(item, bytes):
                    await ws.send_bytes(item)
                else:
                    await ws.send_json(item)

        async def status_pump():
            while True:
                await asyncio.sleep(0.25)
                if session is not None:
                    outbox.put_nowait({"type": "status", **session.status()})

        tasks = [asyncio.create_task(sender())]
        try:
            raw = await ws.receive_json()
            in_sr = int(raw.pop("sample_rate", 48000))
            fields = set(RealtimeStart.model_fields) - {"input_device", "output_device", "monitor_device"}
            cfg = RealtimeConfig(**RealtimeStart(**{k: v for k, v in raw.items() if k in fields}).model_dump(
                exclude={"input_device", "output_device", "monitor_device"}))
            session = BrowserRealtimeSession(
                manager, voices, in_sr, on_audio=lambda pcm: loop.call_soon_threadsafe(outbox.put_nowait, pcm))
            outbox.put_nowait({"type": "loading"})
            await run_in_threadpool(session.start, cfg)  # charge les modèles (peut être long la 1re fois)
            outbox.put_nowait({"type": "started", "out_sample_rate": session.out_sr})
            tasks.append(asyncio.create_task(status_pump()))
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    session.feed(pcm)
                elif msg.get("text") == "stop":
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # erreur de config / modèle : on la renvoie au navigateur
            log.warning("Live navigateur : %s", exc)
            outbox.put_nowait({"type": "error", "detail": str(exc)})
            await asyncio.sleep(0.2)  # laisse le temps au message de partir
        finally:
            for t in tasks:
                t.cancel()
            if session is not None:
                await run_in_threadpool(session.stop)
            try:
                await ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------ interface
    if config.WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")

    return app
