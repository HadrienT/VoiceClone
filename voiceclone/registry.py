"""Catalogue des modèles open source supportés.

Chaque modèle déclare :
- les dépôts Hugging Face à télécharger (avec filtres de fichiers),
- les paquets Python nécessaires (vérifiés à l'exécution, installés à part),
- ses capacités : "tts" (texte -> voix clonée), "vc" (voix -> voix clonée), "asr" (voix -> texte).
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class HFRepo:
    repo_id: str
    allow_patterns: tuple[str, ...] | None = None
    # Sous-dossier local (relatif au dossier du modèle) ; "" = racine
    subdir: str = ""
    revision: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    capabilities: tuple[str, ...]
    engine: str  # "module:Classe" dans voiceclone.engines
    repos: tuple[HFRepo, ...]
    packages: tuple[str, ...]  # modules Python importables requis
    pip: str  # commande d'installation suggérée
    languages: tuple[str, ...] = ()
    license: str = ""
    size_hint: str = ""
    realtime: bool = False  # adapté au mode live
    notes: str = ""
    params: tuple[dict, ...] = field(default=())  # paramètres réglables exposés à l'UI

    def missing_packages(self) -> list[str]:
        return [p for p in self.packages if importlib.util.find_spec(p) is None]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["repos"] = [r["repo_id"] for r in d["repos"]]
        return d


XTTS_LANGS = ("fr", "en", "es", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi")
CHATTERBOX_MTL_LANGS = ("fr", "en", "ar", "da", "de", "el", "es", "fi", "he", "hi", "it", "ja", "ko", "ms", "nl",
                        "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh")
WHISPER_LANGS = ("auto", "fr", "en", "es", "de", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ko", "ar", "tr")

_TEMPERATURE = {"name": "temperature", "label": "Température", "type": "float", "min": 0.1, "max": 1.5, "step": 0.05,
                "default": 0.75, "help": "Plus haut = plus expressif mais moins stable."}
_SPEED = {"name": "speed", "label": "Vitesse", "type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0}
_EXAGGERATION = {"name": "exaggeration", "label": "Expressivité", "type": "float", "min": 0.0, "max": 2.0,
                 "step": 0.05, "default": 0.5, "help": "Intensité émotionnelle (0.5 = neutre)."}
_CFG = {"name": "cfg_weight", "label": "CFG / rythme", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
        "default": 0.5, "help": "Plus bas = débit plus lent et posé. Mettre ~0.3 pour une voix rapide."}

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="xtts-v2",
        name="Coqui XTTS v2",
        description="TTS multilingue de référence pour le clonage zero-shot (6-30 s de voix suffisent). "
                    "Très bon en français, supporte le streaming (faible latence en live).",
        capabilities=("tts",),
        engine="xtts:XTTSEngine",
        repos=(HFRepo("coqui/XTTS-v2", allow_patterns=("config.json", "model.pth", "vocab.json",
                                                       "speakers_xtts.pth", "dvae.pth", "mel_stats.pth")),),
        packages=("TTS", "torch"),
        pip="pip install coqui-tts",
        languages=XTTS_LANGS,
        license="Coqui Public Model License (non commercial)",
        size_hint="≈ 1.9 Go",
        realtime=True,
        params=(_TEMPERATURE, _SPEED,
                {"name": "repetition_penalty", "label": "Pénalité de répétition", "type": "float", "min": 1.0,
                 "max": 20.0, "step": 0.5, "default": 10.0}),
    ),
    ModelSpec(
        id="chatterbox-multilingual",
        name="Chatterbox Multilingual (Resemble AI)",
        description="TTS zero-shot de haute qualité dans 23 langues dont le français, avec contrôle de "
                    "l'expressivité. Licence MIT.",
        capabilities=("tts",),
        engine="chatterbox:ChatterboxMultilingualEngine",
        repos=(HFRepo("ResembleAI/chatterbox", allow_patterns=(
            "ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt", "Cangjie5_TC.json")),),
        packages=("chatterbox", "torch"),
        pip="pip install chatterbox-tts",
        languages=CHATTERBOX_MTL_LANGS,
        license="MIT",
        size_hint="≈ 3.2 Go",
        params=(_EXAGGERATION, _CFG, {**_TEMPERATURE, "default": 0.8}),
    ),
    ModelSpec(
        id="chatterbox",
        name="Chatterbox (anglais) + conversion de voix",
        description="Version anglaise de Chatterbox. Inclut ChatterboxVC : conversion speech-to-speech "
                    "zero-shot (garde vos intonations, change le timbre).",
        capabilities=("tts", "vc"),
        engine="chatterbox:ChatterboxEngine",
        repos=(HFRepo("ResembleAI/chatterbox", allow_patterns=(
            "ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt")),),
        packages=("chatterbox", "torch"),
        pip="pip install chatterbox-tts",
        languages=("en",),
        license="MIT",
        size_hint="≈ 3.0 Go",
        realtime=True,
        params=(_EXAGGERATION, _CFG, {**_TEMPERATURE, "default": 0.8}),
        notes="La conversion de voix (VC) fonctionne quelle que soit la langue parlée.",
    ),
    ModelSpec(
        id="openvoice-v2",
        name="OpenVoice V2 - Tone Color Converter",
        description="Convertisseur de timbre très léger et rapide (MyShell). Idéal pour le speech-to-speech "
                    "en direct sur Discord, même sur une petite carte graphique.",
        capabilities=("vc",),
        engine="openvoice:OpenVoiceEngine",
        repos=(HFRepo("myshell-ai/OpenVoiceV2", allow_patterns=("converter/*",)),),
        packages=("openvoice", "torch"),
        pip="pip install git+https://github.com/myshell-ai/OpenVoice.git",
        languages=("*",),
        license="MIT",
        size_hint="≈ 130 Mo",
        realtime=True,
        params=({"name": "tau", "label": "Tau", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
                 "default": 0.3, "help": "Plus bas = plus proche de la voix cible, plus haut = plus naturel."},),
    ),
    ModelSpec(
        id="f5-tts",
        name="F5-TTS v1 Base",
        description="TTS par flow matching, clonage très fidèle à partir de ~10 s. Optimisé pour anglais "
                    "et chinois. Utilise la transcription de l'audio de référence.",
        capabilities=("tts",),
        engine="f5tts:F5TTSEngine",
        repos=(
            HFRepo("SWivid/F5-TTS", allow_patterns=("F5TTS_v1_Base/model_1250000.safetensors",
                                                    "F5TTS_v1_Base/vocab.txt")),
            HFRepo("charactr/vocos-mel-24khz", allow_patterns=("config.yaml", "pytorch_model.bin"),
                   subdir="vocos"),
        ),
        packages=("f5_tts", "torch"),
        pip="pip install f5-tts",
        languages=("en", "zh"),
        license="CC-BY-NC 4.0 (poids)",
        size_hint="≈ 1.4 Go",
        params=(_SPEED, {"name": "nfe_step", "label": "Étapes (qualité)", "type": "int", "min": 8, "max": 64,
                         "step": 1, "default": 32}),
    ),
    ModelSpec(
        id="whisper-small",
        name="Whisper small (faster-whisper)",
        description="Reconnaissance vocale rapide. Sert au mode live « transcription → TTS » et à "
                    "transcrire automatiquement les échantillons de voix.",
        capabilities=("asr",),
        engine="whisper:WhisperEngine",
        repos=(HFRepo("Systran/faster-whisper-small"),),
        packages=("faster_whisper",),
        pip="pip install faster-whisper",
        languages=WHISPER_LANGS,
        license="MIT",
        size_hint="≈ 480 Mo",
        realtime=True,
    ),
    ModelSpec(
        id="whisper-large-v3-turbo",
        name="Whisper large v3 turbo (faster-whisper)",
        description="Reconnaissance vocale très précise et encore rapide sur GPU.",
        capabilities=("asr",),
        engine="whisper:WhisperEngine",
        repos=(HFRepo("mobiuslabsgmbh/faster-whisper-large-v3-turbo"),),
        packages=("faster_whisper",),
        pip="pip install faster-whisper",
        languages=WHISPER_LANGS,
        license="MIT",
        size_hint="≈ 1.6 Go",
        realtime=True,
    ),
)

_BY_ID = {m.id: m for m in MODELS}

# Modèles additionnels (ex. tests ou extensions) : register_model(spec)
_EXTRA: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    _EXTRA[spec.id] = spec


def unregister_model(model_id: str) -> None:
    _EXTRA.pop(model_id, None)


def all_models() -> list[ModelSpec]:
    return list(MODELS) + list(_EXTRA.values())


def get_model(model_id: str) -> ModelSpec:
    spec = _EXTRA.get(model_id) or _BY_ID.get(model_id)
    if spec is None:
        raise KeyError(f"Modèle inconnu : {model_id}")
    return spec
