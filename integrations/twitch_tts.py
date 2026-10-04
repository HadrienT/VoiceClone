#!/usr/bin/env python3
"""Lecture du chat Twitch avec une voix clonée.

Connexion anonyme en lecture seule (aucun compte ni jeton nécessaire). Chaque message retenu est
envoyé au Live VoiceClone en cours (il part donc vers Discord / OBS comme votre voix), ou
synthétisé et joué sur les haut-parleurs de cette machine avec --play.

    python integrations/twitch_tts.py --channel monpseudo
    python integrations/twitch_tts.py --channel monpseudo --command "!tts" --cooldown 20 --who subs
    python integrations/twitch_tts.py --channel monpseudo --play --voice ma-voix-1a2b3c --model xtts-v2

Variables d'environnement : VOICECLONE_URL (défaut http://127.0.0.1:7860) et VOICECLONE_PASSWORD
si l'interface est protégée.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


@dataclass
class ChatMessage:
    user: str
    text: str
    badges: set[str] = field(default_factory=set)


def parse_line(line: str) -> ChatMessage | None:
    """Analyse une ligne IRC Twitch (avec étiquettes IRCv3) ; None si ce n'est pas un message de chat."""
    tags: dict[str, str] = {}
    if line.startswith("@"):
        raw, _, line = line[1:].partition(" ")
        for kv in raw.split(";"):
            k, _, v = kv.partition("=")
            tags[k] = v
    m = re.match(r":(\w+)!\S+ PRIVMSG #\w+ :(.*)", line)
    if not m:
        return None
    user = tags.get("display-name") or m.group(1)
    badges = {b.split("/")[0] for b in tags.get("badges", "").split(",") if b}
    if tags.get("mod") == "1":
        badges.add("moderator")
    if tags.get("subscriber") == "1":
        badges.add("subscriber")
    return ChatMessage(user=user, text=m.group(2).strip(), badges=badges)


@dataclass
class Policy:
    command: str = ""  # préfixe obligatoire (ex. "!tts") ; vide = tous les messages
    who: str = "all"  # all | subs | vip | mods
    cooldown: float = 15.0  # secondes entre deux messages lus d'une même personne
    max_chars: int = 200
    blocklist: tuple[str, ...] = ()
    ignore_users: tuple[str, ...] = ("nightbot", "streamelements", "streamlabs", "moobot", "fossabot")
    say_name: bool = True
    _last: dict[str, float] = field(default_factory=dict)

    def allowed(self, msg: ChatMessage) -> bool:
        b = msg.badges
        if self.who == "mods":
            return bool(b & {"moderator", "broadcaster"})
        if self.who == "vip":
            return bool(b & {"vip", "moderator", "broadcaster"})
        if self.who == "subs":
            return bool(b & {"subscriber", "founder", "vip", "moderator", "broadcaster"})
        return True

    def text_for(self, msg: ChatMessage, now: float | None = None) -> str | None:
        """Texte à lire pour ce message, ou None s'il est filtré."""
        now = time.time() if now is None else now
        if msg.user.lower() in self.ignore_users or not self.allowed(msg):
            return None
        text = msg.text
        if self.command:
            if not text.lower().startswith(self.command.lower()):
                return None
            text = text[len(self.command):].strip()
        text = URL_RE.sub("un lien", text)
        text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)  # « aaaaaaaa » → « aaa »
        if not text or any(w and w.lower() in text.lower() for w in self.blocklist):
            return None
        if now - self._last.get(msg.user.lower(), 0) < self.cooldown:
            return None
        self._last[msg.user.lower()] = now
        text = text[: self.max_chars]
        return f"{msg.user} dit : {text}" if self.say_name else text


# ------------------------------------------------------------------ VoiceClone
class VoiceCloneClient:
    def __init__(self, url: str, password: str = "") -> None:
        self.url = url.rstrip("/")
        self.password = password

    def _request(self, path: str, payload: dict) -> bytes:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        if self.password:
            req.add_header("Authorization", f"Bearer {self.password}")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail")
            except Exception:
                detail = str(exc)
            raise RuntimeError(detail) from exc

    def say_live(self, text: str) -> None:
        self._request("/api/realtime/say", {"text": text})

    def tts(self, text: str, voice: str, model: str, language: str) -> bytes:
        return self._request("/api/tts", {"text": text, "voice_id": voice, "model_id": model, "language": language})


def play_wav(data: bytes) -> None:
    import io

    import sounddevice as sd
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    sd.play(x, sr)
    sd.wait()


# ------------------------------------------------------------------ Twitch IRC
def connect(channel: str) -> socket.socket:
    raw = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=300)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname="irc.chat.twitch.tv")
    nick = f"justinfan{random.randint(10000, 99999)}"  # connexion anonyme en lecture seule
    for cmd in ("CAP REQ :twitch.tv/tags", "PASS SCHMOOPIIE", f"NICK {nick}", f"JOIN #{channel.lower().lstrip('#')}"):
        sock.sendall((cmd + "\r\n").encode())
    return sock


def run(args) -> None:
    vc = VoiceCloneClient(os.environ.get("VOICECLONE_URL", "http://127.0.0.1:7860"),
                          os.environ.get("VOICECLONE_PASSWORD", ""))
    blocklist = ()
    if args.blocklist:
        with open(args.blocklist, encoding="utf-8") as f:
            blocklist = tuple(w.strip() for w in f if w.strip())
    policy = Policy(command=args.command, who=args.who, cooldown=args.cooldown, max_chars=args.max_chars,
                    blocklist=blocklist, say_name=not args.no_name)
    while True:
        try:
            sock = connect(args.channel)
            print(f"Connecté au chat de #{args.channel} — {'Live VoiceClone' if not args.play else 'haut-parleurs'}")
            buf = ""
            while True:
                data = sock.recv(4096).decode("utf-8", errors="ignore")
                if not data:
                    raise ConnectionError("connexion fermée")
                buf += data
                *lines, buf = buf.split("\r\n")
                for line in lines:
                    if line.startswith("PING"):
                        sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                        continue
                    msg = parse_line(line)
                    text = policy.text_for(msg) if msg else None
                    if not text:
                        continue
                    print(f"🗣  {text}")
                    try:
                        if args.play:
                            play_wav(vc.tts(text, args.voice, args.model, args.language))
                        else:
                            vc.say_live(text)
                    except Exception as exc:
                        print(f"   ⚠ {exc}", file=sys.stderr)
        except (OSError, ConnectionError) as exc:
            print(f"Connexion perdue ({exc}), reconnexion dans 5 s…", file=sys.stderr)
            time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True, help="chaîne Twitch à écouter")
    ap.add_argument("--command", default="", help="préfixe obligatoire, ex. « !tts » (défaut : tous les messages)")
    ap.add_argument("--who", choices=("all", "subs", "vip", "mods"), default="all", help="qui peut se faire lire")
    ap.add_argument("--cooldown", type=float, default=15, help="secondes minimum entre deux messages d'une personne")
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--blocklist", help="fichier texte : un mot interdit par ligne")
    ap.add_argument("--no-name", action="store_true", help="ne pas annoncer « pseudo dit : »")
    ap.add_argument("--play", action="store_true", help="jouer sur cette machine au lieu du Live")
    ap.add_argument("--voice", default="", help="(--play) identifiant de la voix")
    ap.add_argument("--model", default="xtts-v2", help="(--play) modèle TTS")
    ap.add_argument("--language", default="fr", help="(--play) langue")
    args = ap.parse_args()
    if args.play and not args.voice:
        ap.error("--play nécessite --voice")
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
