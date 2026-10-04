"""Mot de passe optionnel sur l'interface et l'API (variable d'environnement VOICECLONE_PASSWORD).

Accès accepté si l'une de ces preuves est présente :
- cookie de session posé par la page de connexion (/login.html) ;
- en-tête « Authorization: Bearer <mot de passe> » (scripts, bot Discord) ou Basic (n'importe quel
  identifiant + le mot de passe) ;
- paramètre « ?key=<mot de passe> » dans l'adresse (sources navigateur OBS) : pose aussi le cookie.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, quote

COOKIE = "voiceclone_auth"
PUBLIC = ("/login.html", "/api/login", "/favicon.ico")


def password() -> str:
    return os.environ.get("VOICECLONE_PASSWORD", "")


def token_for(pw: str) -> str:
    return hmac.new(pw.encode(), b"voiceclone-session", hashlib.sha256).hexdigest()


def check_password(given: str, pw: str) -> bool:
    return bool(pw) and hmac.compare_digest(given.encode(), pw.encode())


class AuthMiddleware:
    def __init__(self, app, pw: str) -> None:
        self.app = app
        self.pw = pw
        self.token = token_for(pw)

    def _authorized(self, scope) -> tuple[bool, bool]:
        """(autorisé, poser le cookie)"""
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        cookie = SimpleCookie(headers.get("cookie", ""))
        if COOKIE in cookie and hmac.compare_digest(cookie[COOKIE].value, self.token):
            return True, False
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer ") and check_password(auth[7:].strip(), self.pw):
            return True, False
        if auth.lower().startswith("basic "):
            try:
                _, _, given = base64.b64decode(auth[6:]).decode().partition(":")
                if check_password(given, self.pw):
                    return True, True
            except Exception:
                pass
        key = parse_qs(scope.get("query_string", b"").decode()).get("key", [""])[0]
        if key and check_password(key, self.pw):
            return True, True
        return False, False

    def cookie_header(self) -> tuple[bytes, bytes]:
        return (b"set-cookie", f"{COOKIE}={self.token}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax".encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if scope["type"] == "http" and path == "/api/login" and scope["method"] == "POST":
            return await self._login(scope, receive, send)
        if scope["type"] == "http" and path == "/api/logout":
            return await self._respond(send, 200, {"ok": True},
                                       [(b"set-cookie", f"{COOKIE}=; Path=/; Max-Age=0".encode())])
        ok, set_cookie = self._authorized(scope)
        if ok or path in PUBLIC:
            if not set_cookie:
                return await self.app(scope, receive, send)

            async def send_with_cookie(msg):
                if msg["type"] == "http.response.start":
                    msg = {**msg, "headers": [*msg.get("headers", []), self.cookie_header()]}
                await send(msg)

            return await self.app(scope, receive, send_with_cookie if scope["type"] == "http" else send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 4401})
        if path.startswith("/api/"):
            return await self._respond(send, 401, {"detail": "Mot de passe requis."},
                                       [(b"www-authenticate", b'Bearer realm="VoiceClone"')])
        target = path + (f"?{scope['query_string'].decode()}" if scope.get("query_string") else "")
        await send({"type": "http.response.start", "status": 302,
                    "headers": [(b"location", f"/login.html?next={quote(target)}".encode())]})
        await send({"type": "http.response.body", "body": b""})

    async def _login(self, scope, receive, send):
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        try:
            given = str(json.loads(body or b"{}").get("password", ""))
        except Exception:
            given = ""
        if not check_password(given, self.pw):
            await asyncio.sleep(1.0)  # freine les essais en rafale
            return await self._respond(send, 401, {"detail": "Mot de passe incorrect."})
        return await self._respond(send, 200, {"ok": True}, [self.cookie_header()])

    @staticmethod
    async def _respond(send, status: int, payload: dict, extra=()):
        body = json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), *extra]})
        await send({"type": "http.response.body", "body": body})
