"""Historique des générations (fichiers WAV + métadonnées)."""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from . import audio, config, settings

MAX_ITEMS = 200


class History:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.OUTPUTS_DIR

    def add(self, wav: np.ndarray, sr: int, **meta) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        item_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        marked = self._save_audio(item_id, wav, sr)
        item = {"id": item_id, "created_at": time.time(), "duration": round(len(wav) / sr, 2),
                **({"watermark": True} if marked else {}), **meta}
        (self.root / f"{item_id}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        self._prune()
        return item

    def _save_audio(self, item_id: str, wav: np.ndarray, sr: int) -> bool:
        marked = bool(settings.get("watermark"))
        if marked:
            from . import watermark

            wav = watermark.embed(wav, sr)
        audio.save_wav(self.root / f"{item_id}.wav", wav, sr)
        return marked

    def get(self, item_id: str) -> dict:
        self.path(item_id)
        return json.loads((self.root / f"{item_id}.json").read_text(encoding="utf-8"))

    def set_meta(self, item_id: str, **fields) -> dict:
        item = {**self.get(item_id), **fields}
        (self.root / f"{item_id}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        return item

    def replace_audio(self, item_id: str, wav: np.ndarray, sr: int, **fields) -> dict:
        self.path(item_id)
        marked = self._save_audio(item_id, wav, sr)
        return self.set_meta(item_id, duration=round(len(wav) / sr, 2), watermark=marked, **fields)

    def parts_dir(self, item_id: str) -> Path:
        """Dossier des phrases d'une génération longue (régénérables une par une)."""
        self.path(item_id)
        return self.root / f"{item_id}.parts"

    def list(self, limit: int = 50) -> list[dict]:
        items = []
        for f in sorted(self.root.glob("*.json"), reverse=True)[:limit]:
            try:
                items.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue
        return items

    def path(self, item_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f-]+", item_id):
            raise KeyError(item_id)
        p = self.root / f"{item_id}.wav"
        if not p.exists():
            raise KeyError(item_id)
        return p

    def update(self, item_id: str, **fields) -> dict:
        self.path(item_id)
        meta = self.root / f"{item_id}.json"
        item = json.loads(meta.read_text(encoding="utf-8"))
        if "favorite" in fields:
            item["favorite"] = bool(fields["favorite"])
        meta.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        return item

    def delete(self, item_id: str) -> None:
        self.path(item_id).unlink(missing_ok=True)
        (self.root / f"{item_id}.json").unlink(missing_ok=True)
        shutil.rmtree(self.root / f"{item_id}.parts", ignore_errors=True)

    def _prune(self) -> None:
        metas = sorted(self.root.glob("*.json"))
        for f in metas[:-MAX_ITEMS]:
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("favorite"):
                    continue  # les favoris ne sont jamais supprimés automatiquement
            except Exception:
                pass
            f.unlink(missing_ok=True)
            f.with_suffix(".wav").unlink(missing_ok=True)
            shutil.rmtree(f.with_suffix(".parts"), ignore_errors=True)
