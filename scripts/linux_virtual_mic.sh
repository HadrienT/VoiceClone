#!/usr/bin/env bash
# Crée un micro virtuel pour Discord sous Linux (PulseAudio ou PipeWire via pipewire-pulse).
#   VoiceClone  -> sortie "VoiceClone_Sink"
#   Discord     -> entrée "VoiceClone_Mic"
# Usage : ./scripts/linux_virtual_mic.sh [remove]
set -euo pipefail

if ! command -v pactl >/dev/null; then
  echo "pactl introuvable : installez pulseaudio-utils (ou pipewire-pulse)." >&2
  exit 1
fi

if [[ "${1:-}" == "remove" ]]; then
  pactl list short modules | awk '/VoiceClone_/ {print $1}' | xargs -r -n1 pactl unload-module
  echo "Périphériques VoiceClone supprimés."
  exit 0
fi

if pactl list short sinks | grep -q VoiceClone_Sink; then
  echo "Le micro virtuel existe déjà."
else
  pactl load-module module-null-sink sink_name=VoiceClone_Sink \
    sink_properties=device.description=VoiceClone_Sink >/dev/null
  pactl load-module module-remap-source master=VoiceClone_Sink.monitor source_name=VoiceClone_Mic \
    source_properties=device.description=VoiceClone_Mic >/dev/null
  echo "Créé !"
fi
cat <<MSG
  • Dans VoiceClone (Live) : sortie = « VoiceClone_Sink » (ou « pulse » puis choisissez-le dans pavucontrol)
  • Dans Discord           : périphérique d'entrée = « VoiceClone_Mic »
Pour supprimer : $0 remove (non persistant après redémarrage).
MSG
