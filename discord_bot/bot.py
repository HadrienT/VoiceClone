"""Bot Discord : fait parler une voix clonée dans un salon vocal.

Commandes (slash) :
  /join            rejoint votre salon vocal
  /leave           quitte le salon
  /voices          liste les voix disponibles
  /say <texte>     lit le texte avec la voix clonée (options : voice, model, language)
  /stop            coupe la lecture en cours

Le bot appelle l'API du serveur VoiceClone, qui doit tourner (python -m voiceclone).
Pour parler *en direct* avec votre micro, utilisez plutôt l'onglet « Live / Discord » de l'interface
avec un câble audio virtuel : un bot Discord ne peut pas remplacer votre propre micro.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import aiohttp
import discord
from discord import app_commands


def _load_env() -> None:
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
API = os.environ.get("VOICECLONE_URL", "http://127.0.0.1:7860").rstrip("/")
DEFAULT_MODEL = os.environ.get("VOICECLONE_MODEL", "xtts-v2")
DEFAULT_VOICE = os.environ.get("VOICECLONE_VOICE", "")
DEFAULT_LANG = os.environ.get("VOICECLONE_LANGUAGE", "fr")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
_queues: dict[int, asyncio.Lock] = {}


async def api_get(path: str):
    async with aiohttp.ClientSession() as s, s.get(API + path) as r:
        r.raise_for_status()
        return await r.json()


async def synthesize(text: str, voice: str, model: str, language: str) -> Path:
    payload = {"model_id": model, "voice_id": voice, "text": text, "language": language}
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as s, s.post(API + "/api/tts", json=payload) as r:
        if r.status != 200:
            try:
                detail = (await r.json()).get("detail")
            except Exception:
                detail = await r.text()
            raise RuntimeError(detail or f"Erreur HTTP {r.status}")
        data = await r.read()
    f = tempfile.NamedTemporaryFile(prefix="voiceclone-", suffix=".wav", delete=False)
    f.write(data)
    f.close()
    return Path(f.name)


async def voice_choices(_: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    try:
        voices = await api_get("/api/voices")
    except Exception:
        return []
    return [app_commands.Choice(name=v["name"][:100], value=v["id"])
            for v in voices if current.lower() in v["name"].lower()][:25]


async def model_choices(_: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    try:
        models = await api_get("/api/models")
    except Exception:
        return []
    return [app_commands.Choice(name=m["name"][:100], value=m["id"]) for m in models
            if "tts" in m["capabilities"] and m["downloaded"] and current.lower() in m["name"].lower()][:25]


async def ensure_connected(inter: discord.Interaction) -> discord.VoiceClient | None:
    vc = inter.guild.voice_client if inter.guild else None
    if vc and vc.is_connected():
        return vc
    member = inter.user if isinstance(inter.user, discord.Member) else None
    if not member or not member.voice or not member.voice.channel:
        return None
    return await member.voice.channel.connect()


@tree.command(description="Rejoindre votre salon vocal")
async def join(inter: discord.Interaction):
    vc = await ensure_connected(inter)
    await inter.response.send_message("🔊 Connecté !" if vc else "Rejoignez d'abord un salon vocal.", ephemeral=True)


@tree.command(description="Quitter le salon vocal")
async def leave(inter: discord.Interaction):
    if inter.guild and inter.guild.voice_client:
        await inter.guild.voice_client.disconnect(force=False)
    await inter.response.send_message("👋", ephemeral=True)


@tree.command(description="Arrêter la lecture en cours")
async def stop(inter: discord.Interaction):
    if inter.guild and inter.guild.voice_client:
        inter.guild.voice_client.stop()
    await inter.response.send_message("⏹️", ephemeral=True)


@tree.command(description="Lister les voix clonées")
async def voices(inter: discord.Interaction):
    try:
        items = await api_get("/api/voices")
    except Exception as exc:
        await inter.response.send_message(f"Serveur VoiceClone injoignable : {exc}", ephemeral=True)
        return
    text = "\n".join(f"• **{v['name']}** (`{v['id']}`, {v['duration']} s)" for v in items) or "Aucune voix."
    await inter.response.send_message(text[:1900], ephemeral=True)


@tree.command(description="Faire parler une voix clonée")
@app_commands.describe(text="Texte à prononcer", voice="Voix", model="Modèle TTS", language="Code langue (fr, en…)")
@app_commands.autocomplete(voice=voice_choices, model=model_choices)
async def say(inter: discord.Interaction, text: str, voice: str | None = None, model: str | None = None,
              language: str | None = None):
    voice = voice or DEFAULT_VOICE
    if not voice:
        try:
            items = await api_get("/api/voices")
            voice = items[0]["id"] if items else ""
        except Exception:
            voice = ""
    if not voice:
        await inter.response.send_message("Aucune voix : créez-en une dans l'interface VoiceClone.", ephemeral=True)
        return
    vc = await ensure_connected(inter)
    if not vc:
        await inter.response.send_message("Rejoignez d'abord un salon vocal.", ephemeral=True)
        return
    await inter.response.defer(thinking=True)
    lock = _queues.setdefault(inter.guild_id, asyncio.Lock())
    async with lock:
        try:
            path = await synthesize(text, voice, model or DEFAULT_MODEL, language or DEFAULT_LANG)
        except Exception as exc:
            await inter.followup.send(f"❌ {exc}")
            return
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        vc.play(discord.FFmpegPCMAudio(str(path)), after=lambda _: loop.call_soon_threadsafe(done.set))
        await inter.followup.send(f"🗣️ {text[:1800]}")
        await done.wait()
        path.unlink(missing_ok=True)


@client.event
async def on_ready():
    await tree.sync()
    print(f"Bot connecté en tant que {client.user} — API VoiceClone : {API}")


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Définissez DISCORD_TOKEN (voir discord_bot/.env.example).")
    client.run(token)


if __name__ == "__main__":
    main()
