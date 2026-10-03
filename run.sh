#!/usr/bin/env bash
# Lance VoiceClone (crée l'environnement virtuel au premier lancement).
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m voiceclone "$@"
