# 🎙️ VoiceClone

Studio **local** de clonage de voix basé sur des modèles **open source**, avec une interface web :

- 📦 **Catalogue de modèles** : choisissez un modèle et téléchargez-le depuis Hugging Face en un clic (barre de progression, suppression, chargement/déchargement de la mémoire GPU).
- 🧬 **Création de voix** : glissez-déposez une ou plusieurs pistes audio **ou enregistrez-vous en direct** depuis le navigateur (texte à lire proposé, vumètre, analyse de qualité). Le bouton « Entraîner » pré-calcule l'empreinte vocale pour un modèle.
- 💬 **Texte → Voix (TTS)** dans la voix clonée.
- 🔁 **Voix → Voix (S2S)** : conversion d'un fichier ou d'un enregistrement.
- 🔴 **Live / Discord** : votre micro est converti en temps réel et envoyé vers un câble audio virtuel que Discord utilise comme micro.
- 🤖 **Bot Discord** optionnel (`/say`) pour faire parler une voix clonée dans un salon.
- 🕘 Historique des générations.

> ⚠️ Ne clonez que votre propre voix ou celle d'une personne qui vous a donné son accord explicite. Usurper l'identité de quelqu'un est illégal dans la plupart des pays.

---

## Modèles supportés

| Modèle | Usage | Français | Live | Licence | Taille |
|---|---|---|---|---|---|
| **Coqui XTTS v2** | TTS | ✅ | ✅ (streaming) | CPML (non commercial) | 1,9 Go |
| **Chatterbox Multilingual** | TTS | ✅ | – | MIT | 3,2 Go |
| **Chatterbox (EN) + VC** | TTS anglais + conversion de voix | VC : ✅ | ✅ | MIT | 3,0 Go |
| **OpenVoice V2** | Conversion de voix (très rapide) | ✅ | ✅ | MIT | 130 Mo |
| **F5-TTS v1** | TTS | ❌ (EN/ZH) | – | CC-BY-NC | 1,4 Go |
| **Whisper small / large-v3-turbo** | Transcription | ✅ | ✅ | MIT | 0,5 / 1,6 Go |

Tous sont **zero-shot** : 10 à 30 s de voix suffisent, pas besoin d'un long entraînement GPU. L'« entraînement » dans l'interface calcule et met en cache l'empreinte vocale (latents XTTS, conditionnements Chatterbox, embedding OpenVoice) pour des générations plus rapides.

Ajouter un modèle = une entrée dans `voiceclone/registry.py` + une classe dans `voiceclone/engines/`.

---

## Installation

Prérequis : **Python 3.10 – 3.11**, `ffmpeg` (recommandé), et de préférence une **carte NVIDIA** (CUDA). Le CPU fonctionne mais le live sera lent.

```bash
git clone https://github.com/HadrienT/VoiceClone.git
cd VoiceClone
python -m venv .venv
# Windows : .venv\Scripts\activate    Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

> Si votre `python3` système est plus récent (3.12+) ou sans module `venv` (Debian), utilisez [uv](https://docs.astral.sh/uv/) : `uv venv --python 3.11 .venv` puis `uv pip install -r requirements.txt` (et `uv pip install …` à la place de `pip install …` ci-dessous). `./run.sh` le fait automatiquement si `uv` est installé.

### 1. PyTorch (avec CUDA)

Installez PyTorch adapté à votre GPU depuis <https://pytorch.org/get-started/locally/>, par exemple :

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Les moteurs que vous voulez utiliser

Chaque modèle a ses propres dépendances ; l'interface affiche la commande à lancer si elles manquent.

```bash
pip install coqui-tts                 # XTTS v2  (recommandé pour le français)
pip install faster-whisper            # Whisper  (mode « transcription + TTS », transcription auto)
pip install git+https://github.com/myshell-ai/OpenVoice.git   # OpenVoice V2 (live rapide)
pip install chatterbox-tts 'setuptools<81'   # Chatterbox (TTS multilingue + VC)
pip install f5-tts                    # F5-TTS
```

> 💡 Ces paquets épinglent parfois des versions différentes de `torch`/`transformers`. Si vous rencontrez un conflit, créez un environnement virtuel par moteur (ou commencez par **XTTS + Whisper + OpenVoice**, combo qui couvre TTS, S2S et live en français). Après une installation pip, **relancez le serveur**.

### 3. Audio temps réel

`sounddevice` nécessite PortAudio : inclus sous Windows/macOS, sous Linux : `sudo apt install libportaudio2`.

---

## Lancement

```bash
python -m voiceclone            # ouvre http://localhost:7860
# ou : ./run.sh   /   run.bat   (crée le venv au premier lancement)
```

Options : `--host 0.0.0.0` (accès depuis le réseau local), `--port 7860`, `--no-browser`.

Variables d'environnement :

| Variable | Rôle | Défaut |
|---|---|---|
| `VOICECLONE_DATA` | dossier des modèles, voix et générations | `./data` |
| `VOICECLONE_DEVICE` | `auto`, `cuda`, `cuda:1`, `mps`, `cpu` | `auto` |
| `HF_TOKEN` | jeton Hugging Face (dépôts privés / quotas) | – |
| `VOICECLONE_MAX_REF_SECONDS` | durée max de la référence vocale | `30` |

> L'enregistrement micro dans le navigateur n'est autorisé que sur `localhost` ou en HTTPS.

Avec plusieurs GPU, `auto` choisit celui qui a le plus de mémoire libre au démarrage.

### Sur un serveur distant (SSH)

Le navigateur n'est pas ouvert automatiquement sans affichage. Depuis votre poste, ouvrez un tunnel puis allez sur <http://localhost:7860> (le micro du navigateur fonctionne ainsi, contrairement à un accès par l'adresse IP) :

```bash
ssh -L 7860:localhost:7860 utilisateur@serveur
```

L'onglet **Live / Discord**, en mode « Audio de ce PC » (par défaut), capte le micro du PC où la page est ouverte, envoie l'audio au serveur par WebSocket (via le tunnel), et rejoue la voix convertie sur la sortie choisie de ce PC (ex. *CABLE Input*). Le mode « Audio du serveur » n'est utile que si VoiceClone tourne sur votre propre PC.

---

## Utilisation

1. **Modèles** → *Télécharger* (ex. XTTS v2 et OpenVoice V2). Optionnel : *Charger en mémoire* pour éviter l'attente à la première génération.
2. **Mes voix** → donnez un nom, glissez vos fichiers ou cliquez **● Enregistrer** et lisez le texte proposé (2-3 prises de 10 s). Cochez le consentement → *Créer la voix*. Vérifiez les avertissements qualité (bruit, saturation…), puis *🧬 Entraîner* pour le modèle choisi.
3. **Texte → Voix** : choisissez modèle, voix, langue, tapez le texte → *Générer*.
4. **Voix → Voix** : *Conversion directe* (garde votre intonation) ou *Transcription + TTS* (re-synthèse totale).

### Enregistrer le son du PC (ce que vous entendez au casque)

Dans **Mes voix** (et **Voix → Voix**), le menu de source propose, en plus des micros, **🖥️ Son du PC** : il enregistre
ce qui sort de l'ordinateur sur lequel la page est ouverte (vidéo YouTube, appel, jeu…). La capture se fait **dans le
navigateur**, donc ça fonctionne même quand VoiceClone tourne sur un serveur distant.

1. Choisissez **🖥️ Son du PC** puis **● Enregistrer**.
2. Windows (Chrome ou Edge) : dans la fenêtre de partage, onglet **« Écran entier »**, cochez **« Partager aussi l'audio
   du système »** puis *Partager*. Pour ne capter qu'un onglet (ex. une vidéo), choisissez l'onglet et cochez
   « Partager aussi l'audio de l'onglet ».
3. **■ Arrêter** (ou « Arrêter le partage » dans la barre du navigateur) : l'échantillon est ajouté.

L'image n'est jamais enregistrée. Firefox et Safari ne savent pas capturer le son système. Évitez la musique de fond,
et ne clonez que des voix pour lesquelles vous avez l'accord de la personne.

### ✨ Import intelligent (recommandé)

Activé par défaut dans **Mes voix** (et via le bouton ✨ de chaque voix) : déposez un enregistrement brut,
VoiceClone s'occupe du reste.

1. **Nettoyage** : coupe des basses inutiles, réduction du bruit de fond stationnaire (souffle, ventilateur)
   seulement si nécessaire, volume égalisé.
2. **Découpe** en phrases naturelles (pauses ≥ 250 ms), morceaux de 3 à 11 s.
3. **Notation** de chaque morceau : rapport voix/bruit mesuré *dans* la phrase, part de parole, saturation, durée.
4. **Sélection** des meilleurs jusqu'à la durée choisie (30 s par défaut), le meilleur en premier
   (c'est lui que Chatterbox utilise). Les passages trop bruités, saturés ou sans parole sont écartés.
5. **Transcription** des passages gardés si un Whisper est téléchargé (utile pour F5-TTS).

Un rapport indique pour chaque passage s'il a été gardé et pourquoi. En ligne de commande :
`python scripts/add_samples.py "Ma voix" mon_audio.wav --auto [--target 30] [--no-denoise]`.

Limites : le débruitage vise les bruits constants ; il ne retire ni la musique ni une autre voix
(ces passages sont en revanche souvent écartés par la notation).

### Conseils pour un bon clone

- 10 à 30 s de parole naturelle, **une seule personne**, sans musique ni écho.
- Micro proche, gain sans saturation, pièce calme.
- Variez les intonations (questions, exclamations).
- Pour F5-TTS, renseignez la transcription exacte (ou bouton *Transcrire avec Whisper*).

---

## 🎮 Utiliser la voix clonée en direct sur Discord

Le principe : VoiceClone lit votre **vrai micro**, convertit la voix, et joue le résultat dans un **câble audio virtuel**. Discord écoute ce câble comme s'il s'agissait d'un micro.

```
Votre micro ──► VoiceClone (modèle) ──► câble virtuel ──► Discord (entrée)
                         └──► casque (retour optionnel)
```

1. Installez un câble virtuel :
   - **Windows** : [VB-CABLE](https://vb-audio.com/Cable/) → sortie *CABLE Input*, Discord entrée *CABLE Output*.
   - **macOS** : [BlackHole 2ch](https://existential.audio/blackhole/) → sortie et entrée *BlackHole 2ch*.
   - **Linux** : `./scripts/linux_virtual_mic.sh` → sortie *VoiceClone_Sink*, Discord entrée *VoiceClone_Mic*
     (avec PortAudio/ALSA, choisissez la sortie `pulse` puis redirigez le flux vers *VoiceClone_Sink* dans `pavucontrol`).
2. Onglet **Live / Discord**, mode **« Audio de ce PC »** (dans Chrome ou Edge) : micro en entrée, câble en sortie (marqué ★), casque en « Retour casque » si vous voulez vous entendre. Le calcul reste sur le serveur ; seul l'audio transite par le navigateur.
3. Testez d'abord le mode **Test du routage** (micro brut) pour vérifier que Discord vous entend.
4. Dans Discord : désactivez **Krisp / suppression de bruit** et la **sensibilité automatique** (ou baissez le seuil).

### Modes live

| Mode | Fonctionnement | Latence typique (GPU) | Modèles |
|---|---|---|---|
| **Conversion directe** | Votre voix découpée en morceaux (≈ 0,7 s) avec contexte et fondu enchaîné | 0,5 – 1 s | OpenVoice V2, Chatterbox VC |
| **Transcription + TTS** | Détection de fin de phrase → Whisper → TTS en streaming | 1 – 2 s après la fin de phrase | Whisper + XTTS (streaming), Chatterbox… |

Réglages : taille des morceaux (plus petit = moins de latence mais plus d'artefacts), contexte, seuil de silence (le silence n'est pas envoyé au modèle), gains. Si « morceaux sautés » augmente, le GPU ne suit pas : augmentez la taille des morceaux ou prenez OpenVoice.

### Bot Discord (texte → voix dans un salon)

```bash
pip install -r discord_bot/requirements.txt
cp discord_bot/.env.example discord_bot/.env   # renseignez DISCORD_TOKEN
python discord_bot/bot.py
```

Créez l'application sur <https://discord.com/developers/applications>, invitez le bot avec les scopes `bot` + `applications.commands` et les permissions *Connect* / *Speak*. Commandes : `/join`, `/say texte [voice] [model] [language]`, `/voices`, `/stop`, `/leave`. Nécessite `ffmpeg`.

---

## API

L'interface s'appuie sur une API REST documentée automatiquement sur <http://localhost:7860/docs>. Principaux points d'entrée :

| Méthode | Route | Description |
|---|---|---|
| GET | `/api/models` | catalogue + état (téléchargé, dépendances, chargé, progression) |
| POST | `/api/models/{id}/download` · `/load` · `/unload` | gestion des modèles |
| GET/POST | `/api/voices` | liste / création (multipart : `name`, `files[]`, `consent=true`) |
| POST | `/api/voices/{id}/samples` · `/prepare` · `/transcribe` | échantillons, entraînement, transcription |
| POST | `/api/tts` | `{model_id, voice_id, text, language, params}` → WAV |
| POST | `/api/vc` | multipart `file`, `model_id`, `voice_id`, `mode` → WAV |
| GET/POST | `/api/realtime/devices` · `/start` · `/stop` · `/status` | live |

Exemple :

```bash
curl -X POST localhost:7860/api/tts -H 'Content-Type: application/json' \
  -d '{"model_id":"xtts-v2","voice_id":"ma-voix-1a2b3c","text":"Salut tout le monde !","language":"fr"}' \
  -o sortie.wav
```

---

## Structure

```
voiceclone/
  server.py      API FastAPI + service de l'interface web
  registry.py    catalogue des modèles (dépôts HF, dépendances, paramètres)
  downloads.py   téléchargements Hugging Face en arrière-plan
  manager.py     chargement/déchargement des moteurs, verrou GPU
  voices.py      profils de voix (nettoyage audio, référence, cache)
  realtime.py    moteur live (micro → modèle → câble virtuel)
  audio.py       utilitaires audio (décodage, rééchantillonnage, analyse)
  engines/       XTTS, Chatterbox, OpenVoice, F5-TTS, Whisper
web/             interface (HTML/CSS/JS, sans build)
discord_bot/     bot Discord optionnel
scripts/         micro virtuel Linux
tests/           tests (moteur factice, pas de téléchargement)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Les tests utilisent un moteur factice : ils vérifient l'API, la gestion des voix, les téléchargements (simulés) et les boucles temps réel sans GPU ni modèle.

## Dépannage

- **« Dépendances manquantes »** : lancez la commande `pip install` affichée, puis redémarrez le serveur.
- **CUDA out of memory** : déchargez les modèles inutilisés (onglet Modèles) ; XTTS ≈ 3 Go VRAM, Chatterbox ≈ 5-6 Go, OpenVoice < 1 Go.
- **« Audio temps réel indisponible »** : installez PortAudio (`libportaudio2`).
- **Discord coupe la voix** : désactivez Krisp et la sensibilité automatique.
- **Voix robotique en live** : augmentez la taille des morceaux et le contexte, vérifiez que le seuil de silence ne coupe pas vos fins de phrases.
- **Fichier refusé** : installez `ffmpeg` pour les formats exotiques.
- **`libnvrtc.so.13` / erreur `torchcodec`** : depuis PyTorch 2.9, `torchaudio` passe par `torchcodec`, qui doit être compilé pour la même version de CUDA que `torch`. VoiceClone lit l'audio de référence sans `torchcodec`, mais d'autres bibliothèques peuvent encore en avoir besoin. Pour réparer : `python -c "import torch; print(torch.__version__, torch.version.cuda)"`, puis réinstallez `torchcodec` depuis le même index que torch, par ex. `pip install --force-reinstall torchcodec --index-url https://download.pytorch.org/whl/cu128` (remplacez `cu128` par votre version de CUDA).
- **F5-TTS : « besoin du texte exact »** : F5 doit connaître le texte prononcé dans l'échantillon. Renseignez-le dans *Mes voix → Échantillons & transcription*, ou téléchargez un modèle Whisper pour qu'il soit transcrit automatiquement (une seule fois, mis en cache). F5 n'utilise qu'un échantillon de 12 s maximum : pour un long enregistrement, découpez-le aux silences avec `python scripts/split_audio.py mon_audio.wav` et importez les morceaux dans la même voix. Si le fichier est déjà sur le serveur, `python scripts/add_samples.py "Ma voix" mon_audio.wav --split` le découpe et l'ajoute directement à la voix (sans passer par le navigateur ; `--list` affiche les voix, `--create NOM --consent` en crée une).
