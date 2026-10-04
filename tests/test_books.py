import io
import time
import zipfile

from tests.conftest import make_voice_wav
from voiceclone.books import parse_epub, parse_text, parse_upload

EPUB_OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Mon Livre</dc:title></metadata>
 <manifest>
  <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
  <item id="c1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>
  <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
 </manifest>
 <spine><itemref idref="cover"/><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"""


def make_epub() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf", EPUB_OPF)
        z.writestr("OEBPS/cover.xhtml", "<html><body><img src='c.jpg'/></body></html>")
        z.writestr("OEBPS/text/ch1.xhtml", "<html><head><title>x</title><style>p{}</style></head><body>"
                   "<h1>Le début</h1><p>Il était une fois un petit village tranquille.</p><p>Fin du chapitre un.</p></body></html>")
        z.writestr("OEBPS/text/ch2.xhtml", "<html><body><h2>La suite</h2><p>Le lendemain, tout avait changé "
                   "dans le village et personne ne savait pourquoi.</p></body></html>")
    return buf.getvalue()


def test_parse_text_chapters_and_size_split():
    chs = parse_text("Préface courte.\n\nChapitre 1 : Départ\nBonjour.\n\nChapitre 2 : Arrivée\nAu revoir.")
    assert [c["title"] for c in chs] == ["Introduction", "Chapitre 1 : Départ", "Chapitre 2 : Arrivée"]
    md = parse_text("# Un\ntexte un\n\n# Deux\ntexte deux")
    assert [c["title"] for c in md] == ["Un", "Deux"] and md[1]["text"] == "texte deux"
    big = parse_text("\n\n".join(["Phrase assez longue pour remplir. " * 30] * 20))
    assert len(big) >= 2 and big[0]["title"] == "Partie 1"


def test_parse_epub():
    title, chs = parse_epub(make_epub())
    assert title == "Mon Livre"
    assert [c["title"] for c in chs] == ["Le début", "La suite"]
    assert "petit village" in chs[0]["text"] and "Le début" not in chs[0]["text"]
    assert parse_upload("x.epub", make_epub())["chapters"][0]["chars"] > 10


def test_book_generation_end_to_end(client, fake_model):
    v = client.post("/api/voices", data={"name": "V", "consent": "true"},
                    files=[("files", ("a.wav", make_voice_wav(4.0), "audio/wav"))]).json()
    parsed = client.post("/api/books/parse", files={"file": ("livre.epub", make_epub(), "application/epub+zip")}).json()
    r = client.post("/api/books", json={"title": parsed["title"], "chapters": parsed["chapters"],
                                         "model_id": "fake", "voice_id": v["id"], "format": "mp3"})
    assert r.status_code == 200, r.text
    book_id, job_id = r.json()["book"]["id"], r.json()["job"]["id"]
    for _ in range(300):
        if client.get(f"/api/jobs/{job_id}").json()["state"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    b = client.get(f"/api/books/{book_id}").json()
    assert b["state"] == "done", b
    assert all(c["file"].endswith(".mp3") for c in b["chapters"])
    ch = client.get(f"/api/books/{book_id}/chapters/0")
    assert ch.status_code == 200 and ch.headers["content-type"] == "audio/mpeg"
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/books/{book_id}/zip").content))
    assert z.namelist() == ["001 - Le début.mp3", "002 - La suite.mp3"]
    assert client.get("/api/books").json()[0]["id"] == book_id
    client.delete(f"/api/books/{book_id}")
    assert client.get(f"/api/books/{book_id}").status_code == 404
