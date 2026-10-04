"""Application FastAPI : API REST + interface web."""

from __future__ import annotations

import asyncio
import json
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

from . import __version__, audio, auth, config, diagnostics, logs, opus, settings
from . import device as devmod
from .downloads import DownloadManager
from .engines.base import EngineError
from .history import History
from .jobs import JobManager
from .manager import EngineManager, ModelNotReady
from .realtime import (
    BrowserRealtimeSession,
    RealtimeConfig,
    RealtimeSession,
    RealtimeUnavailable,
    list_devices,
)
from .registry import get_model
from .voices import VoiceStore, consent_record

log = logging.getLogger("voiceclone")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


# ---------------------------------------------------------------- schémas
class TTSRequest(BaseModel):
    model_id: str
    voice_id: str
    text: str = Field(min_length=1, max_length=5000)
    language: str = "fr"
    params: dict = Field(default_factory=dict)


class LongTTSRequest(TTSRequest):
    text: str = Field(min_length=1, max_length=200_000)
    title: str = ""


class BookChapter(BaseModel):
    title: str = Field("", max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class BookRequest(BaseModel):
    title: str = Field("Livre audio", max_length=200)
    chapters: list[BookChapter] = Field(min_length=1, max_length=500)
    model_id: str
    voice_id: str
    language: str = "fr"
    params: dict = Field(default_factory=dict)
    format: str = "mp3"  # mp3 | wav
    announce_titles: bool = True


class SegmentRegen(BaseModel):
    text: str | None = Field(None, max_length=2000)
    params: dict | None = None


class VoiceUpdate(BaseModel):
    name: str | None = None
    language: str | None = None
    description: str | None = None
    transcript: str | None = None
    settings: dict | None = None


class MixSource(BaseModel):
    voice_id: str
    weight: float = Field(ge=0, le=100)


class MixRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    sources: list[MixSource] = Field(min_length=2, max_length=5)
    language: str | None = None


class SampleOrder(BaseModel):
    files: list[str]  # noms des échantillons (ex. "003.wav") dans l'ordre voulu


class HistoryUpdate(BaseModel):
    favorite: bool | None = None


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
    say_model_id: str | None = None
    warmup: bool = True
    translate_to: str | None = None
    mt_model_id: str | None = None


class SayRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class SettingsUpdate(BaseModel):
    max_loaded_models: int | None = Field(None, ge=0, le=20)
    auto_unload: bool | None = None
    watermark: bool | None = None
    engine_python: dict[str, str] | None = None


# ------------------------------------------------------------------- app
def create_app() -> FastAPI:
    config.ensure_dirs()
    logs.install()
    downloads = DownloadManager()
    manager = EngineManager(downloads)
    voices = VoiceStore()
    history = History()
    live = RealtimeSession(manager, voices)
    jobs = JobManager()

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
    app.state.jobs = jobs

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

    @app.get("/api/diagnostics")
    def diag():
        return diagnostics.report(manager)

    @app.get("/api/logs")
    def get_logs(after: int = 0, level: str = "INFO"):
        return {"last": logs.HANDLER.counter, "records": logs.HANDLER.since(after, level)}

    @app.get("/api/settings")
    def get_settings():
        return settings.load()

    @app.patch("/api/settings")
    def patch_settings(body: SettingsUpdate):
        return settings.update(**body.model_dump(exclude_none=True))

    # -------------------------------------------------------------- tâches
    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return jobs.get(job_id).to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id).to_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        jobs.remove(job_id)
        return {"ok": True}

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

    @app.post("/api/models/unload-idle")
    def unload_idle():
        """Libère la mémoire : décharge tous les modèles qui ne servent pas à un Live en cours."""
        done = [m["id"] for m in manager.loaded() if not m["pinned"]]
        for mid in done:
            manager.unload(mid)
        return {"unloaded": done}

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
        consent_owner: str = Form("self"),
        transcript: str = Form(""),
        source: str = Form("upload"),
        files: list[UploadFile] = File(default=[]),
    ):
        if not consent:
            raise HTTPException(400, "Vous devez confirmer avoir le droit d'utiliser cette voix.")
        payloads = [(await read_upload(f), f.filename or "") for f in files]
        voice = voices.create(name, language, description, consent=consent_record(consent_owner, "web"))
        try:
            for data, fname in payloads:
                voice = voices.add_sample(voice.id, data, source=source, original_name=fname,
                                          transcript=transcript if len(payloads) == 1 else "")
        except Exception:
            voices.delete(voice.id)
            raise
        return voice.to_dict()

    @app.post("/api/voices/mix")
    def mix_voices(body: MixRequest):
        """Crée une voix intermédiaire (ex. 70 % voix A + 30 % voix B)."""
        try:
            return voices.create_mix(body.name, [(s.voice_id, s.weight) for s in body.sources], body.language).to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

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
        method: str = Form("auto"),
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
            pieces, report = auto_prepare(x, sr, enhance=enhance, method=method,
                                          target_s=max(5.0, min(target_seconds, 120.0)))
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

    @app.get("/api/enhance/methods")
    def enhance_methods():
        from . import enhance

        return enhance.describe()

    @app.put("/api/voices/{voice_id}/order")
    def reorder_samples(voice_id: str, body: SampleOrder):
        """Ordre des échantillons : le premier sert de référence principale (Chatterbox n'en écoute que 10 s)."""
        return voices.reorder(voice_id, [f"samples/{f}" for f in body.files]).to_dict()

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
                           params=req.params,
                           text=req.text[:500], language=req.language, seconds=round(elapsed, 2))
        # fichier enregistré (avec le filigrane s'il est activé)
        return Response(history.path(item["id"]).read_bytes(), media_type="audio/wav", headers={
            "X-History-Id": item["id"], "X-Generation-Seconds": f"{elapsed:.2f}",
            "X-Audio-Seconds": f"{len(wav) / sr:.2f}",
        })

    def render_long(job, req: LongTTSRequest, kind: str = "long", extra: dict | None = None) -> dict:
        """Génère un texte long phrase par phrase (tâche de fond) ; chaque phrase reste régénérable."""
        from . import longform

        voice = voices.get(req.voice_id)
        segments = longform.parse(req.text)
        texts = [i for i, sg in enumerate(segments) if sg["type"] == "text"]
        if not texts:
            raise ValueError("Aucun texte à lire.")
        job.update(0.0, "Chargement du modèle…")
        engine = manager.get(req.model_id, "tts")
        wavs: dict[int, np.ndarray] = {}
        sr = 24000
        t0 = time.time()
        for n, i in enumerate(texts, 1):
            job.update((n - 1) / len(texts), f"Phrase {n}/{len(texts)}")
            with manager.infer_lock:
                wav, sr = engine.tts(segments[i]["text"], voice, req.language, **req.params)
            wavs[i] = wav
        full = longform.assemble([(sg, wavs.get(i)) for i, sg in enumerate(segments)], sr)
        item = history.add(full, sr, kind=kind, model_id=req.model_id, voice_id=voice.id, voice_name=voice.name,
                           params=req.params, text=req.text[:500], language=req.language,
                           seconds=round(time.time() - t0, 2), title=req.title or None, **(extra or {}))
        parts = history.parts_dir(item["id"])
        parts.mkdir(parents=True, exist_ok=True)
        for i, wav in wavs.items():
            audio.save_wav(parts / f"{i:04d}.wav", wav, sr)
            segments[i]["file"] = f"{i:04d}.wav"
            segments[i]["duration"] = round(len(wav) / sr, 2)
        history.set_meta(item["id"], segments=segments, sample_rate=sr)
        return {"history_id": item["id"], "duration": item["duration"]}

    @app.post("/api/tts/long")
    def tts_long(req: LongTTSRequest):
        """Texte long (ou avec des [pause]) : tâche de fond avec progression ; voir /api/jobs/{id}."""
        voices.get(req.voice_id)
        get_model(req.model_id)
        title = req.title or (req.text[:40] + ("…" if len(req.text) > 40 else ""))
        return jobs.submit("tts", title, lambda job: render_long(job, req)).to_dict()

    @app.get("/api/history/{item_id}/segments/{index}/audio")
    def segment_audio(item_id: str, index: int):
        path = history.parts_dir(item_id) / f"{index:04d}.wav"
        if not path.exists():
            raise KeyError("Phrase introuvable")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/history/{item_id}/segments/{index}")
    def regenerate_segment(item_id: str, index: int, body: SegmentRegen):
        """Régénère une seule phrase (texte éventuellement corrigé) puis recolle l'ensemble."""
        from . import longform

        item = history.get(item_id)
        segments = item.get("segments") or []
        if not (0 <= index < len(segments)) or segments[index]["type"] != "text":
            raise KeyError("Phrase introuvable")
        voice = voices.get(item["voice_id"])
        engine = manager.get(item["model_id"], "tts")
        text = (body.text or segments[index]["text"]).strip()
        params = body.params if body.params is not None else item.get("params") or {}
        with manager.infer_lock:
            wav, sr = engine.tts(text, voice, item.get("language", "fr"), **params)
        parts = history.parts_dir(item_id)
        target_sr = item.get("sample_rate", sr)
        if sr != target_sr:
            wav = audio.resample(wav, sr, target_sr)
        audio.save_wav(parts / f"{index:04d}.wav", wav, target_sr)
        segments[index].update(text=text, file=f"{index:04d}.wav", duration=round(len(wav) / target_sr, 2),
                               regenerated=segments[index].get("regenerated", 0) + 1)
        loaded = [(sg, audio.load_audio(parts / sg["file"])[0] if sg.get("file") else None) for sg in segments]
        full = longform.assemble(loaded, target_sr)
        return history.replace_audio(item_id, full, target_sr, segments=segments,
                                     text=" ".join(sg.get("text", "") for sg in segments if sg["type"] == "text")[:500])

    # ------------------------------------------------------------ livres audio
    from .books import BookStore, build_zip, parse_upload, write_chapter

    books = BookStore()

    @app.post("/api/books/parse")
    async def book_parse(file: UploadFile = File(...)):
        """Découpe un .txt / .md / .epub en chapitres (à relire avant de lancer la génération)."""
        data = await read_upload(file)
        try:
            return parse_upload(file.filename or "livre.txt", data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/books")
    def book_create(req: BookRequest):
        from . import longform

        voice = voices.get(req.voice_id)
        get_model(req.model_id)
        book_id = books.new_id()
        meta = {"id": book_id, "title": req.title, "created_at": time.time(), "model_id": req.model_id,
                "voice_id": voice.id, "voice_name": voice.name, "language": req.language, "state": "queued",
                "chapters": [{"title": c.title or f"Chapitre {i}", "chars": len(c.text)}
                             for i, c in enumerate(req.chapters, 1)]}
        books.save(book_id, meta)

        def work(job):
            engine = manager.get(req.model_id, "tts")
            plans = []
            for c in req.chapters:
                head = f"{c.title}. [pause 1s]\n\n" if req.announce_titles and c.title else ""
                plans.append(longform.parse(head + c.text))
            total = sum(1 for p in plans for sg in p if sg["type"] == "text") or 1
            done = 0
            meta["state"] = "running"
            t0 = time.time()
            for ci, (plan, ch) in enumerate(zip(plans, meta["chapters"])):
                wavs, sr = {}, 24000
                for i, sg in enumerate(plan):
                    if sg["type"] != "text":
                        continue
                    job.update(done / total, f"Chapitre {ci + 1}/{len(plans)} · phrase {done + 1}/{total}")
                    with manager.infer_lock:
                        wavs[i], sr = engine.tts(sg["text"], voice, req.language, **req.params)
                    done += 1
                full = longform.assemble([(sg, wavs.get(i)) for i, sg in enumerate(plan)], sr)
                if settings.get("watermark"):
                    from . import watermark

                    full = watermark.embed(full, sr)
                path = write_chapter(books.root / book_id / f"{ci + 1:03d}", full, sr, req.format)
                ch.update(file=path.name, duration=round(len(full) / sr, 2))
                books.save(book_id, meta)
            meta["zip"] = build_zip(books.root / book_id, meta)
            meta.update(state="done", seconds=round(time.time() - t0, 1),
                        duration=round(sum(c.get("duration", 0) for c in meta["chapters"]), 1))
            books.save(book_id, meta)
            return {"book_id": book_id, "duration": meta["duration"]}

        def guarded(job):
            try:
                return work(job)
            except BaseException as exc:
                meta["state"] = "cancelled" if job.cancelled else "error"
                meta["error"] = str(exc) or type(exc).__name__
                books.save(book_id, meta)
                raise

        job = jobs.submit("book", f"📚 {req.title}", guarded)
        meta["job_id"] = job.id
        books.save(book_id, meta)
        return {"book": meta, "job": job.to_dict()}

    def book_state(meta: dict) -> dict:
        if meta.get("state") in ("queued", "running") and meta.get("job_id"):
            try:
                job = jobs.get(meta["job_id"])
                meta["progress"] = job.progress
                if job.state in ("cancelled", "error"):
                    meta["state"] = job.state
            except KeyError:  # serveur redémarré pendant la génération
                meta["state"] = "error"
                meta["error"] = "Génération interrompue (redémarrage du serveur)."
        return meta

    @app.get("/api/books")
    def book_list():
        return [book_state(b) for b in books.list()]

    @app.get("/api/books/{book_id}")
    def book_get(book_id: str):
        return book_state(books.get(book_id))

    @app.get("/api/books/{book_id}/chapters/{index}")
    def book_chapter(book_id: str, index: int):
        path = books.chapter_path(book_id, index)
        return FileResponse(path, media_type="audio/mpeg" if path.suffix == ".mp3" else "audio/wav",
                            filename=path.name)

    @app.get("/api/books/{book_id}/zip")
    def book_zip(book_id: str):
        path = books.zip_path(book_id)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.delete("/api/books/{book_id}")
    def book_delete(book_id: str):
        meta = books.get(book_id)
        if meta.get("job_id") and meta.get("state") in ("queued", "running"):
            try:
                jobs.cancel(meta["job_id"])
            except KeyError:
                pass
        books.delete(book_id)
        return {"ok": True}

    # ------------------------------------------------------------ filigrane
    @app.post("/api/watermark/detect")
    async def watermark_detect(file: UploadFile = File(...)):
        from starlette.concurrency import run_in_threadpool

        from . import watermark

        data = await read_upload(file)

        def work():
            x, sr = audio.load_audio(data)
            return watermark.detect(x, sr)

        return await run_in_threadpool(work)

    # ----------------------------------------------------- speech-to-speech
    @app.post("/api/vc")
    async def vc(
        file: UploadFile = File(...),
        model_id: str = Form(...),
        voice_id: str = Form(...),
        mode: str = Form("vc"),  # vc | asr_tts | translate
        asr_model_id: str = Form(""),
        language: str = Form("fr"),
        source_voice_id: str = Form(""),
        params: str = Form("{}"),
        target_language: str = Form(""),
        mt_model_id: str = Form(""),
    ):
        import json

        from starlette.concurrency import run_in_threadpool

        data = await read_upload(file)
        extra = json.loads(params or "{}")

        def work():
            voice = voices.get(voice_id)
            x, sr = audio.load_audio(data)
            t0 = time.time()
            text = translated = None
            if mode == "translate":
                from .translation import speech_to_translated_text

                if not asr_model_id or not target_language:
                    raise HTTPException(400, "Choisissez un modèle de transcription et la langue d'arrivée.")
                asr = manager.get(asr_model_id, "asr")
                tts_engine = manager.get(model_id, "tts")
                text, translated = speech_to_translated_text(manager, asr, x, sr, language, target_language,
                                                             mt_model_id or None)
                if not translated:
                    raise HTTPException(400, "Aucune parole détectée dans l'audio.")
                with manager.infer_lock:
                    wav, out_sr = tts_engine.tts(translated, voice, target_language, **extra)
            elif mode == "asr_tts":
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
            return wav, out_sr, text, translated, time.time() - t0, voice

        wav, sr, text, translated, elapsed, voice = await run_in_threadpool(work)
        item = history.add(wav, sr, kind="translate" if mode == "translate" else "s2s", mode=mode,
                           model_id=model_id, voice_id=voice.id, voice_name=voice.name,
                           text=(translated or text or "")[:500], source_text=(text or "")[:500] if translated else None,
                           language=target_language or language, seconds=round(elapsed, 2))
        headers = {"X-History-Id": item["id"], "X-Generation-Seconds": f"{elapsed:.2f}"}
        if text:
            headers["X-Transcript"] = quote(text[:500])
        if translated:
            headers["X-Translation"] = quote(translated[:500])
        return Response(history.path(item["id"]).read_bytes(), media_type="audio/wav", headers=headers)

    # ------------------------------------------------------------ historique
    @app.get("/api/history")
    def list_history(limit: int = 50):
        return history.list(limit)

    @app.get("/api/history/{item_id}")
    def get_history(item_id: str):
        return history.get(item_id)

    @app.get("/api/history/{item_id}/audio")
    def history_audio(item_id: str, format: str = "wav"):
        path = history.path(item_id)
        if format == "mp3":
            return Response(audio.encode_mp3(path), media_type="audio/mpeg",
                            headers={"Content-Disposition": f'attachment; filename="voiceclone-{item_id}.mp3"'})
        return FileResponse(path, media_type="audio/wav", filename=f"voiceclone-{item_id}.wav")

    @app.patch("/api/history/{item_id}")
    def update_history(item_id: str, body: HistoryUpdate):
        return history.update(item_id, **body.model_dump(exclude_none=True))

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

    @app.post("/api/realtime/say")
    def rt_say(body: SayRequest):
        live.say(body.text)
        return {"ok": True}

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

        async def handle_control(sess, ctrl: dict, box: asyncio.Queue):
            kind = ctrl.get("type")
            if kind == "silence":  # micro coupé / porte de bruit : du silence sans le transmettre
                sess.feed(np.zeros(max(0, min(int(ctrl.get("n", 0)), sess.in_sr)), dtype=np.float32))
            elif kind == "ping":  # mesure de l'aller-retour réseau par le navigateur
                box.put_nowait({"type": "pong", "t": ctrl.get("t")})
            elif kind == "say":
                try:
                    sess.say(ctrl.get("text", ""))
                except Exception as exc:
                    box.put_nowait({"type": "notice", "detail": str(exc)})

        tasks = [asyncio.create_task(sender())]
        try:
            raw = await ws.receive_json()
            in_sr = int(raw.pop("sample_rate", 48000))
            # Opus si le navigateur le propose, que PyAV est installé et que la fréquence s'y prête
            use_opus = raw.pop("codec", "pcm") == "opus" and opus.available() and in_sr in opus.OPUS_RATES
            decoder = opus.OpusDecoder(in_sr) if use_opus else None
            encoder = opus.OpusEncoder(48000) if use_opus else None

            def on_audio(y: np.ndarray) -> None:  # appelé depuis le thread de traitement
                if encoder is not None:
                    for packet in encoder.encode(y):
                        loop.call_soon_threadsafe(outbox.put_nowait, packet)
                else:
                    loop.call_soon_threadsafe(outbox.put_nowait, (y * 32767).astype("<i2").tobytes())

            fields = set(RealtimeStart.model_fields) - {"input_device", "output_device", "monitor_device"}
            cfg = RealtimeConfig(**RealtimeStart(**{k: v for k, v in raw.items() if k in fields}).model_dump(
                exclude={"input_device", "output_device", "monitor_device"}))
            session = BrowserRealtimeSession(manager, voices, in_sr, on_audio=on_audio,
                                             out_sr=48000 if use_opus else None)
            outbox.put_nowait({"type": "loading"})
            await run_in_threadpool(session.start, cfg)  # charge les modèles (peut être long la 1re fois)
            outbox.put_nowait({"type": "started", "out_sample_rate": session.out_sr,
                               "codec": "opus" if use_opus else "pcm"})
            tasks.append(asyncio.create_task(status_pump()))
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    if decoder is not None:
                        session.feed(decoder.decode(msg["bytes"]))
                    else:
                        session.feed(np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0)
                elif msg.get("text") == "stop":
                    break
                elif msg.get("text"):
                    await handle_control(session, json.loads(msg["text"]), outbox)
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
    @app.get("/api/auth")
    def auth_state():
        return {"protected": bool(auth.password())}

    if config.WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")

    if auth.password():  # protège interface, API et WebSocket
        app.add_middleware(auth.AuthMiddleware, pw=auth.password())
    return app
