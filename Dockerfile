# VoiceClone — image prête à l'emploi (GPU NVIDIA ou CPU).
#
#   docker build -t voiceclone .                                     # CUDA 12.4 + XTTS, Whisper, OpenVoice
#   docker build -t voiceclone --build-arg TORCH=cpu --build-arg MODELS="whisper-small openvoice-v2" .
#   docker run --gpus all -p 7860:7860 -v voiceclone-data:/app/data voiceclone
#
# Les poids des modèles se téléchargent depuis l'interface et restent dans le volume /app/data.
# Les bibliothèques CUDA sont fournies par les roues PyTorch : une image Python légère suffit,
# avec le NVIDIA Container Toolkit installé sur l'hôte.
FROM python:3.11-slim

ARG TORCH=cu124
ARG MODELS="xtts-v2 whisper-small openvoice-v2"

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VOICECLONE_DATA=/app/data HF_HOME=/app/data/hf-cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libportaudio2 libsndfile1 git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY voiceclone ./voiceclone
COPY scripts ./scripts
COPY web ./web
RUN if [ -n "$MODELS" ]; then python scripts/install.py --torch "$TORCH" $MODELS; \
    elif [ -n "$TORCH" ]; then pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$TORCH"; fi

COPY . .
VOLUME /app/data
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/system')" || exit 1
CMD ["python", "-m", "voiceclone", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
