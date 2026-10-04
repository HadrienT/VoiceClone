"""Livres audio : découpage d'un texte / .txt / .md / .epub en chapitres, génération chapitre par chapitre.

Arborescence : data/outputs/books/<id>/
    book.json        titre, voix, modèle, chapitres (titre, durée, fichier)
    001.mp3 …        un fichier par chapitre (WAV si ffmpeg est absent)
    <titre>.zip      archive de tous les chapitres
"""

from __future__ import annotations

import io
import json
import posixpath
import re
import shutil
import time
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from . import audio, config

CHAPTER_RE = re.compile(r"^\s*(?:#{1,3}\s+.+|(?:chapitre|chapter|partie|part|livre|book|prologue|épilogue|epilogue)\b.{0,80})$",
                        re.I | re.M)
MAX_CHAPTER_CHARS = 12000


# ------------------------------------------------------------------ analyse
def _by_size(text: str, prefix: str = "Partie") -> list[dict]:
    paras = re.split(r"\n\s*\n", text.strip())
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > MAX_CHAPTER_CHARS:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        out.append(cur)
    return [{"title": f"{prefix} {i}", "text": t} for i, t in enumerate(out, 1)]


def parse_text(text: str) -> list[dict]:
    """Chapitres d'un texte brut / Markdown : titres « Chapitre … », « # … » ; sinon découpe par taille."""
    text = text.replace("\r\n", "\n").strip()
    marks = list(CHAPTER_RE.finditer(text))
    if not marks:
        return _by_size(text)
    chapters = []
    if marks[0].start() > 0 and text[: marks[0].start()].strip():
        chapters.append({"title": "Introduction", "text": text[: marks[0].start()].strip()})
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end(): end].strip()
        title = m.group(0).strip().lstrip("#").strip()
        if body:
            chapters.append({"title": title[:120], "text": body})
    return chapters


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "li", "section", "blockquote", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_h = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        if tag in ("h1", "h2", "h3") and not self.title:
            self._in_h = True
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        if tag in ("h1", "h2", "h3"):
            self._in_h = False
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_h:
            self.title += data
        self.parts.append(data)

    def text(self) -> str:
        t = re.sub(r"[ \t\xa0]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", t).strip()


def parse_epub(data: bytes) -> tuple[str, list[dict]]:
    """(titre, chapitres) d'un EPUB, dans l'ordre de lecture (spine)."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        container = ElementTree.fromstring(z.read("META-INF/container.xml"))
    except Exception as exc:
        raise ValueError(f"EPUB illisible : {exc}") from exc
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container", "opf": "http://www.idpf.org/2007/opf",
          "dc": "http://purl.org/dc/elements/1.1/"}
    opf_path = container.find(".//c:rootfile", ns).get("full-path")
    opf = ElementTree.fromstring(z.read(opf_path))
    base = posixpath.dirname(opf_path)
    title = (opf.findtext(".//dc:title", default="", namespaces=ns) or "Livre").strip()
    manifest = {i.get("id"): i.get("href") for i in opf.findall(".//opf:manifest/opf:item", ns)}
    chapters = []
    for ref in opf.findall(".//opf:spine/opf:itemref", ns):
        href = manifest.get(ref.get("idref"))
        if not href:
            continue
        try:
            raw = z.read(posixpath.normpath(posixpath.join(base, href))).decode("utf-8", errors="ignore")
        except KeyError:
            continue
        p = _HTMLText()
        p.feed(raw)
        body = p.text()
        if len(body) < 40:  # couverture, pages vides
            continue
        name = (p.title.strip() or f"Chapitre {len(chapters) + 1}")[:120]
        # retire le titre répété en tête du texte
        if body.startswith(name):
            body = body[len(name):].strip()
        chapters.append({"title": name, "text": body})
    return title, chapters


def parse_upload(filename: str, data: bytes) -> dict:
    name = Path(filename or "livre").stem
    if filename.lower().endswith(".epub"):
        title, chapters = parse_epub(data)
    else:
        text = data.decode("utf-8", errors="ignore")
        if filename.lower().endswith((".html", ".htm", ".xhtml")):
            p = _HTMLText()
            p.feed(text)
            text = p.text()
        title, chapters = name, parse_text(text)
    for c in chapters:
        c["chars"] = len(c["text"])
    return {"title": title or name, "chapters": chapters}


# ------------------------------------------------------------------ stockage
class BookStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.OUTPUTS_DIR / "books"

    def dir(self, book_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f-]+", book_id) or not (self.root / book_id / "book.json").exists():
            raise KeyError(f"Livre introuvable : {book_id}")
        return self.root / book_id

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    def save(self, book_id: str, meta: dict) -> dict:
        d = self.root / book_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "book.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return meta

    def get(self, book_id: str) -> dict:
        return json.loads((self.dir(book_id) / "book.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        out = []
        for f in sorted(self.root.glob("*/book.json"), reverse=True):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def delete(self, book_id: str) -> None:
        shutil.rmtree(self.dir(book_id), ignore_errors=True)

    def chapter_path(self, book_id: str, index: int) -> Path:
        meta = self.get(book_id)
        ch = meta["chapters"][index]
        if not ch.get("file"):
            raise KeyError("Chapitre pas encore généré")
        return self.dir(book_id) / ch["file"]

    def zip_path(self, book_id: str) -> Path:
        meta = self.get(book_id)
        return self.dir(book_id) / meta["zip"]


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\- ]+", "", s, flags=re.UNICODE).strip()[:60] or "livre"


def write_chapter(path_noext: Path, wav, sr: int, fmt: str) -> Path:
    wav_path = path_noext.with_suffix(".wav")
    audio.save_wav(wav_path, wav, sr)
    if fmt == "mp3" and shutil.which("ffmpeg"):
        mp3 = path_noext.with_suffix(".mp3")
        mp3.write_bytes(audio.encode_mp3(wav_path, "128k"))
        wav_path.unlink(missing_ok=True)
        return mp3
    return wav_path


def build_zip(book_dir: Path, meta: dict) -> str:
    name = f"{safe_name(meta['title'])}.zip"
    with zipfile.ZipFile(book_dir / name, "w", compression=zipfile.ZIP_STORED) as z:
        for i, ch in enumerate(meta["chapters"], 1):
            if ch.get("file"):
                z.write(book_dir / ch["file"], f"{i:03d} - {safe_name(ch['title'])}{Path(ch['file']).suffix}")
    return name
