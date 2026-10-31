# VoiceClone

A local voice cloning studio built on open-source models, with a web interface.
It runs on your own machine: models come from Hugging Face, and audio stays on
the server.

![Python 3.10-3.11](https://img.shields.io/badge/Python-3.10--3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)

> Only clone your own voice, or the voice of someone who has given explicit
> consent. Each voice stores a dated consent declaration, and generated audio
> can carry an inaudible watermark with a matching detector.

The interface and the full user guide are in French:
[`docs/guide.fr.md`](docs/guide.fr.md).

## Features

- **Model catalogue**: download, load and unload models from the interface, with
  VRAM monitoring and automatic eviction of the least recently used model.
- **Voice creation**: upload, record from the browser, or capture system audio.
  Smart import cleans a raw recording, splits it into phrases, scores each one
  and keeps the best.
- **Text to speech**: per-voice settings, side-by-side model comparison, and
  long texts generated sentence by sentence, each one fixable on its own.
- **Audiobooks**: `.txt`, `.md` or `.epub` to one MP3 per chapter.
- **Voice conversion and translation**: convert a file or a recording, or speak
  French and have your cloned voice speak English.
- **Live**: real-time conversion into a virtual audio cable for Discord or OBS,
  with push-to-talk, noise gate, Opus compression and auto-reconnect. A Discord
  bot and a Twitch chat reader are included.
- **Training**: fine-tune XTTS on a voice, or import or train RVC models.
- **Voice mixing**: blend two voices at a chosen ratio.

## Supported models

| Model | Purpose | French | Live | License | Size |
|---|---|---|---|---|---|
| Coqui XTTS v2 | TTS | Yes | Yes (streaming) | CPML (non-commercial) | 1.9 GB |
| Chatterbox Multilingual | TTS | Yes | No | MIT | 3.2 GB |
| Chatterbox (EN) + VC | English TTS, voice conversion | VC only | Yes | MIT | 3.0 GB |
| OpenVoice V2 | Voice conversion (very fast) | Yes | Yes | MIT | 130 MB |
| F5-TTS v1 | TTS | No (EN/ZH) | No | CC-BY-NC | 1.4 GB |
| Seed-VC | Voice conversion (diffusion) | Yes | Yes | GPL-3.0 (code) | 1.6 GB |
| RVC | Conversion with a per-voice trained model | Yes | Yes | MIT | 370 MB + model |
| Whisper small / large-v3-turbo | Transcription | Yes | Yes | MIT | 0.5 / 1.6 GB |
| Opus-MT fr-en / en-fr | Translation | Yes | Yes | CC-BY | 300 MB |
| NLLB-200 600M | Translation (200 languages) | Yes | No | CC-BY-NC | 2.5 GB |

All voice models except RVC are zero-shot: 10 to 30 seconds of speech is enough.
Adding a model means one entry in `voiceclone/registry.py` and one class in
`voiceclone/engines/`.

## Getting started

Requires Python 3.10 or 3.11, `ffmpeg`, and preferably an NVIDIA GPU. The CPU
works, but live conversion will be slow.

```bash
git clone https://github.com/HadrienT/VoiceClone.git
cd VoiceClone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# PyTorch for your CUDA version, then the engines you want
python scripts/install.py --list
python scripts/install.py --torch cu124 xtts-v2 whisper-small openvoice-v2

python -m voiceclone    # http://localhost:7860
```

`./run.sh` and `run.bat` create the environment on first launch. With Docker:
`docker compose up -d` (requires the NVIDIA Container Toolkit).

Engines often pin incompatible versions of `torch`, `transformers` or `numpy`.
`python scripts/install.py --isolated <model>` installs one in its own virtual
environment, and VoiceClone runs it in a separate process.

| Variable | Purpose | Default |
|---|---|---|
| `VOICECLONE_DATA` | Models, voices and generated audio | `./data` |
| `VOICECLONE_DEVICE` | `auto`, `cuda`, `cuda:1`, `mps`, `cpu` | `auto` |
| `VOICECLONE_PASSWORD` | Protects the interface, API and WebSocket | none |
| `HF_TOKEN` | Hugging Face token | none |

## API

The interface runs on a REST API, documented at `http://localhost:7860/docs`.

```bash
curl -X POST localhost:7860/api/tts -H 'Content-Type: application/json' \
  -d '{"model_id":"xtts-v2","voice_id":"my-voice-1a2b3c","text":"Hello!","language":"en"}' \
  -o output.wav
```

## Project layout

| Path | Contents |
|---|---|
| `voiceclone/` | FastAPI server, model manager, voice profiles, real-time engine, training |
| `voiceclone/engines/` | One adapter per model family |
| `web/` | Interface in plain HTML, CSS and JavaScript, with no build step |
| `discord_bot/`, `integrations/` | Discord bot, Twitch chat reader |
| `scripts/` | Installer, real-model smoke test, audio tools |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses a fake engine, so it needs neither a GPU nor a model download.
`python scripts/smoke_test.py` exercises the models actually installed on a
machine. Model-dependent features still awaiting validation on a real GPU are
tracked in [`WORKPLAN.md`](WORKPLAN.md).

## License

The code is released under the [MIT License](LICENSE). It does not cover the third-party
models, which keep their own licenses (see [Supported models](#supported-models)).
