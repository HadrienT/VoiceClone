"""Bot Discord : fait parler des voix clonées dans un salon vocal.

Commandes (slash) :
  /join              rejoint votre salon vocal
  /leave             quitte le salon
  /voices            liste les voix disponibles
  /say <texte>       lit le texte (options : voice, model, language) — mis en file d'attente
  /mavoix <voix>     choisit VOTRE voix par défaut (et modèle / langue) pour /say et /lire
  /lire on|off       lit à voix haute les messages écrits dans ce salon textuel, chacun avec sa voix
  /file              affiche la file d'attente
  /skip              passe la phrase en cours
  /stop              coupe tout et vide la file
  /live <texte>      envoie le texte au Live VoiceClone en cours (sort comme votre propre micro)

Le bot appelle l'API du serveur VoiceClone, qui doit tourner (python -m voiceclone).
Pour parler *en direct* avec votre micro, utilisez l'onglet « Live / Discord » de l'interface avec
un câble audio virtuel : un bot Discord ne peut pas remplacer votre propre micro.

/lire nécessite l'intention privilégiée « Message Content » (portail développeur Discord → Bot) et
DISCORD_READ_MESSAGES=1 dans le .env.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass
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
PASSWORD = os.environ.get("VOICECLONE_PASSWORD", "")
DEFAULT_MODEL = os.environ.get("VOICECLONE_MODEL", "xtts-v2")
DEFAULT_VOICE = os.environ.get("VOICECLONE_VOICE", "")
DEFAULT_LANG = os.environ.get("VOICECLONE_LANGUAGE", "fr")
READ_MESSAGES = os.environ.get("DISCORD_READ_MESSAGES", "0") == "1"
MAX_QUEUE = int(os.environ.get("DISCORD_MAX_QUEUE", "20"))
PREFS_FILE = Path(__file__).with_name("prefs.json")

intents = discord.Intents.default()
intents.message_content = READ_MESSAGES
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# ------------------------------------------------------------------ préférences
def load_prefs() -> dict:
    try:
        return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"users": {}, "read_channels": []}


def save_prefs() -> None:
    PREFS_FILE.write_text(json.dumps(PREFS, ensure_ascii=False, indent=1), encoding="utf-8")


PREFS = load_prefs()
PREFS.setdefault("users", {})
PREFS.setdefault("read_channels", [])


def user_pref(user_id: int) -> dict:
    return PREFS["users"].get(str(user_id), {})


# ------------------------------------------------------------------ API VoiceClone
def _headers() -> dict:
    return {"Authorization": f"Bearer {PASSWORD}"} if PASSWORD else {}


async def _error(r: aiohttp.ClientResponse) -> str:
    try:
        return (await r.json()).get("detail") or f"Erreur HTTP {r.status}"
    except Exception:
        return await r.text() or f"Erreur HTTP {r.status}"


async def api_get(path: str):
    async with aiohttp.ClientSession(headers=_headers()) as s, s.get(API + path) as r:
        if r.status != 200:
            raise RuntimeError(await _error(r))
        return await r.json()


async def api_post(path: str, payload: dict) -> bytes:
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout, headers=_headers()) as s, s.post(API + path, json=payload) as r:
        if r.status != 200:
            raise RuntimeError(await _error(r))
        return await r.read()


async def synthesize(text: str, voice: str, model: str, language: str) -> Path:
    data = await api_post("/api/tts", {"model_id": model, "voice_id": voice, "text": text, "language": language})
    f = tempfile.NamedTemporaryFile(prefix="voiceclone-", suffix=".wav", delete=False)
    f.write(data)
    f.close()
    return Path(f.name)


async def resolve_voice(user_id: int, voice: str | None) -> str:
    voice = voice or user_pref(user_id).get("voice") or DEFAULT_VOICE
    if voice:
        return voice
    items = await api_get("/api/voices")
    return items[0]["id"] if items else ""


# ------------------------------------------------------------------ file de lecture
@dataclass
class Item:
    text: str
    voice: str
    model: str
    language: str
    author: str


class GuildPlayer:
    """Une file par serveur Discord : synthèse puis lecture, une phrase après l'autre."""

    def __init__(self, guild: discord.Guild) -> None:
        self.guild = guild
        self.queue: asyncio.Queue[Item] = asyncio.Queue(maxsize=MAX_QUEUE)
        self.current: Item | None = None
        self.task = asyncio.create_task(self._run())

    def pending(self) -> list[Item]:
        return list(self.queue._queue)  # noqa: SLF001 - simple aperçu

    def clear(self) -> None:
        while not self.queue.empty():
            self.queue.get_nowait()

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            vc = self.guild.voice_client
            if not vc or not vc.is_connected():
                continue
            self.current = item
            path = None
            try:
                path = await synthesize(item.text, item.voice, item.model, item.language)
                done = asyncio.Event()
                loop = asyncio.get_running_loop()
                vc.play(discord.FFmpegPCMAudio(str(path)), after=lambda _: loop.call_soon_threadsafe(done.set))
                await done.wait()
            except Exception as exc:
                print(f"Lecture impossible : {exc}")
            finally:
                self.current = None
                if path:
                    path.unlink(missing_ok=True)


PLAYERS: dict[int, GuildPlayer] = {}


def player(guild: discord.Guild) -> GuildPlayer:
    if guild.id not in PLAYERS:
        PLAYERS[guild.id] = GuildPlayer(guild)
    return PLAYERS[guild.id]


async def enqueue(guild: discord.Guild, item: Item) -> int:
    p = player(guild)
    try:
        p.queue.put_nowait(item)
    except asyncio.QueueFull as exc:
        raise RuntimeError(f"File pleine ({MAX_QUEUE} phrases) : attendez un peu.") from exc
    return p.queue.qsize()


# ------------------------------------------------------------------ autocomplétion
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


# ------------------------------------------------------------------ commandes
@tree.command(description="Rejoindre votre salon vocal")
async def join(inter: discord.Interaction):
    vc = await ensure_connected(inter)
    await inter.response.send_message("🔊 Connecté !" if vc else "Rejoignez d'abord un salon vocal.", ephemeral=True)


@tree.command(description="Quitter le salon vocal")
async def leave(inter: discord.Interaction):
    if inter.guild and inter.guild.voice_client:
        player(inter.guild).clear()
        await inter.guild.voice_client.disconnect(force=False)
    await inter.response.send_message("👋", ephemeral=True)


@tree.command(description="Passer la phrase en cours")
async def skip(inter: discord.Interaction):
    if inter.guild and inter.guild.voice_client:
        inter.guild.voice_client.stop()
    await inter.response.send_message("⏭️", ephemeral=True)


@tree.command(description="Tout arrêter et vider la file d'attente")
async def stop(inter: discord.Interaction):
    if inter.guild:
        player(inter.guild).clear()
        if inter.guild.voice_client:
            inter.guild.voice_client.stop()
    await inter.response.send_message("⏹️ File vidée.", ephemeral=True)


@tree.command(name="file", description="Voir la file d'attente")
async def show_queue(inter: discord.Interaction):
    p = player(inter.guild)
    lines = ([f"▶️ **{p.current.author}** : {p.current.text[:80]}"] if p.current else []) + \
        [f"{i}. **{it.author}** : {it.text[:80]}" for i, it in enumerate(p.pending(), 1)]
    await inter.response.send_message("\n".join(lines)[:1900] or "File vide.", ephemeral=True)


@tree.command(description="Lister les voix clonées")
async def voices(inter: discord.Interaction):
    try:
        items = await api_get("/api/voices")
    except Exception as exc:
        await inter.response.send_message(f"Serveur VoiceClone injoignable : {exc}", ephemeral=True)
        return
    mine = user_pref(inter.user.id).get("voice")
    text = "\n".join(f"• **{v['name']}** (`{v['id']}`, {v['duration']} s){' ⭐' if v['id'] == mine else ''}"
                     for v in items) or "Aucune voix."
    await inter.response.send_message(text[:1900], ephemeral=True)


@tree.command(description="Choisir votre voix par défaut (pour /say et la lecture du salon)")
@app_commands.describe(voice="Voix", model="Modèle TTS", language="Code langue (fr, en…)")
@app_commands.autocomplete(voice=voice_choices, model=model_choices)
async def mavoix(inter: discord.Interaction, voice: str, model: str | None = None, language: str | None = None):
    pref = {"voice": voice, **({"model": model} if model else {}), **({"language": language} if language else {})}
    PREFS["users"][str(inter.user.id)] = {**user_pref(inter.user.id), **pref}
    save_prefs()
    await inter.response.send_message("✅ Voix enregistrée pour vous.", ephemeral=True)


@tree.command(description="Faire parler une voix clonée dans le salon vocal")
@app_commands.describe(text="Texte à prononcer", voice="Voix", model="Modèle TTS", language="Code langue (fr, en…)")
@app_commands.autocomplete(voice=voice_choices, model=model_choices)
async def say(inter: discord.Interaction, text: str, voice: str | None = None, model: str | None = None,
              language: str | None = None):
    try:
        voice = await resolve_voice(inter.user.id, voice)
    except Exception as exc:
        await inter.response.send_message(f"Serveur VoiceClone injoignable : {exc}", ephemeral=True)
        return
    if not voice:
        await inter.response.send_message("Aucune voix : créez-en une dans l'interface VoiceClone.", ephemeral=True)
        return
    if not await ensure_connected(inter):
        await inter.response.send_message("Rejoignez d'abord un salon vocal.", ephemeral=True)
        return
    pref = user_pref(inter.user.id)
    try:
        pos = await enqueue(inter.guild, Item(text, voice, model or pref.get("model") or DEFAULT_MODEL,
                                              language or pref.get("language") or DEFAULT_LANG,
                                              inter.user.display_name))
    except RuntimeError as exc:
        await inter.response.send_message(f"❌ {exc}", ephemeral=True)
        return
    await inter.response.send_message(f"🗣️ {text[:1800]}" + (f"  *(position {pos} dans la file)*" if pos > 1 else ""))


@tree.command(description="Envoyer un texte au Live VoiceClone en cours (sort comme votre micro)")
async def live(inter: discord.Interaction, text: str):
    try:
        await api_post("/api/realtime/say", {"text": text})
        await inter.response.send_message("📡 Envoyé au Live.", ephemeral=True)
    except Exception as exc:
        await inter.response.send_message(f"❌ {exc}", ephemeral=True)


@tree.command(description="Lire à voix haute les messages de ce salon textuel (on/off)")
@app_commands.choices(etat=[app_commands.Choice(name="on", value="on"), app_commands.Choice(name="off", value="off")])
async def lire(inter: discord.Interaction, etat: app_commands.Choice[str]):
    if not READ_MESSAGES:
        await inter.response.send_message("Activez DISCORD_READ_MESSAGES=1 et l'intention « Message Content » "
                                          "du bot pour utiliser cette commande.", ephemeral=True)
        return
    chans = set(PREFS["read_channels"])
    if etat.value == "on":
        if not await ensure_connected(inter):
            await inter.response.send_message("Rejoignez d'abord un salon vocal.", ephemeral=True)
            return
        chans.add(inter.channel_id)
    else:
        chans.discard(inter.channel_id)
    PREFS["read_channels"] = sorted(chans)
    save_prefs()
    await inter.response.send_message(f"📖 Lecture du salon : **{etat.value}**.")


def clean_message(msg: discord.Message) -> str:
    text = msg.clean_content
    text = re.sub(r"https?://\S+", "un lien", text)
    text = re.sub(r"<a?:(\w+):\d+>", r"\1", text)  # émojis personnalisés
    return text.strip()[:300]


@client.event
async def on_message(msg: discord.Message):
    if msg.author.bot or not msg.guild or msg.channel.id not in PREFS["read_channels"]:
        return
    vc = msg.guild.voice_client
    text = clean_message(msg)
    if not vc or not vc.is_connected() or not text or text.startswith("/"):
        return
    pref = user_pref(msg.author.id)
    try:
        voice = await resolve_voice(msg.author.id, None)
        if voice:
            await enqueue(msg.guild, Item(text, voice, pref.get("model") or DEFAULT_MODEL,
                                          pref.get("language") or DEFAULT_LANG, msg.author.display_name))
    except Exception as exc:
        print(f"Message ignoré : {exc}")


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
