"""Historique des générations (fichiers WAV + métadonnées)."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from . import audio, config

MAX_ITEMS = 200


class History:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.OUTPUTS_DIR

    def add(self, wav: np.ndarray, sr: int, **meta) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        item_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        audio.save_wav(self.root / f"{item_id}.wav", wav, sr)
        item = {"id": item_id, "created_at": time.time(), "duration": round(len(wav) / sr, 2), **meta}
        (self.root / f"{item_id}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        self._prune()
        return item

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

    def delete(self, item_id: str) -> None:
        self.path(item_id).unlink(missing_ok=True)
        (self.root / f"{item_id}.json").unlink(missing_ok=True)

    def _prune(self) -> None:
        metas = sorted(self.root.glob("*.json"))
        for f in metas[:-MAX_ITEMS]:
            f.unlink(missing_ok=True)
            f.with_suffix(".wav").unlink(missing_ok=True)
