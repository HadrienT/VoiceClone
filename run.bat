@echo off
REM Lance VoiceClone (crée l'environnement virtuel au premier lancement).
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python -m voiceclone %*
