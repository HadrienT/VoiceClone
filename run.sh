#!/usr/bin/env bash
# Lance VoiceClone (crée l'environnement virtuel au premier lancement).
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  # Les moteurs demandent Python 3.10-3.11 : uv le fournit même si le python3 système est plus récent
  # (ou livré sans le module venv/pip, comme sous Debian).
  if command -v uv >/dev/null; then
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi
exec .venv/bin/python -m voiceclone "$@"
