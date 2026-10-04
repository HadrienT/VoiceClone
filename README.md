# 🎙️ VoiceClone

Studio **local** de clonage de voix basé sur des modèles **open source**, avec une interface web :

- 📦 **Catalogue de modèles** : choisissez un modèle et téléchargez-le depuis Hugging Face en un clic (progression, suppression, chargement / déchargement de la mémoire GPU).
- 🧬 **Création de voix** : glissez-déposez des pistes audio, **enregistrez-vous** depuis le navigateur ou **capturez le son du PC**. ✂ Forme d'onde pour couper avant l'import, ✨ **import intelligent** (nettoyage, découpe, sélection des meilleurs passages ; DeepFilterNet / Demucs en option), ordre des échantillons, transcription Whisper, 🎛 **mélange de voix**.
- 💬 **Texte → Voix** : réglages mémorisés par voix, ⚖ comparateur de modèles, **textes longs** générés phrase par phrase (pauses `[pause 1s]`, chaque phrase corrigeable seule).
- 📚 **Livre audio** : texte, `.txt`, `.md` ou `.epub` → un MP3 par chapitre + archive ZIP.
- 🔁 **Voix → Voix** : conversion d'un fichier ou d'un enregistrement, et 🌍 **traduction vocale** (vous parlez français, votre voix clonée parle anglais).
- 🔴 **Live / Discord** : votre micro converti en temps réel vers un câble audio virtuel (Discord, OBS…). Appuyer pour parler, porte de bruit, compression Opus, reconnexion auto, texte tapé dit par la voix clonée, traduction en direct, page de sous-titres pour OBS, lecture du chat Twitch.
- 🎓 **Entraînement** : affiner XTTS sur votre voix, modèles RVC (import ou entraînement via Applio).
- 🤖 **Bot Discord** : file d'attente, une voix par membre, lecture d'un salon textuel.
- 🩺 **Diagnostic** : GPU et mémoire, versions, journal du serveur, réglages (déchargement automatique, filigrane, isolation des moteurs).
- 🔐 Mot de passe optionnel, consentement enregistré avec chaque voix, filigrane inaudible et détecteur.
- 🕘 Historique (favoris, filtres, MP3, régénérer).

Le détail des fonctions et de ce qui reste à valider sur un vrai GPU est dans [WORKPLAN.md](WORKPLAN.md).

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
| **Seed-VC** | Conversion de voix (diffusion, haute fidélité) | ✅ | ✅ (4-8 étapes) | GPL-3.0 (code) | 1,6 Go |
| **RVC** | Conversion avec un modèle entraîné par voix | ✅ | ✅ | MIT | 370 Mo + modèle |
| **Whisper small / large-v3-turbo** | Transcription | ✅ | ✅ | MIT | 0,5 / 1,6 Go |
| **Opus-MT fr→en / en→fr** | Traduction | ✅ | ✅ | CC-BY | 300 Mo |
| **NLLB-200 600M** | Traduction (200 langues) | ✅ | – | CC-BY-NC | 2,5 Go |

Tous les modèles de voix sauf RVC sont **zero-shot** : 10 à 30 s de voix suffisent. Le bouton « Préparer » d'une voix calcule et met en cache son empreinte (latents XTTS, conditionnements Chatterbox, timbre OpenVoice, référence Seed-VC) pour des générations plus rapides. Pour aller plus loin, l'onglet **Entraînement** affine XTTS sur votre voix ou produit un modèle RVC.

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
pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git && pip install librosa   # OpenVoice V2 (live rapide)
pip install chatterbox-tts 'setuptools<81'   # Chatterbox (TTS multilingue + VC)
pip install f5-tts                    # F5-TTS
```

Ou avec le script d'installation (lit les commandes du catalogue) :

```bash
python scripts/install.py --list                         # modèles et état des dépendances
python scripts/install.py --torch cu124 xtts-v2 whisper-small openvoice-v2
python scripts/install.py --isolated rvc                 # venv dédié data/envs/rvc + réglage automatique
```

> 💡 Ces paquets épinglent parfois des versions incompatibles de `torch`/`transformers`/`numpy` (RVC, Seed-VC surtout). `--isolated` installe un modèle dans son propre environnement : VoiceClone le fait alors tourner dans un processus séparé (réglable aussi dans **Diagnostic → Réglages → Isolation**). Commencez par **XTTS + Whisper + OpenVoice**, combo qui couvre TTS, S2S et live en français. Après une installation pip, **relancez le serveur**.

Optionnels : `pip install av` (compression Opus du Live), `pip install deepfilternet` / `pip install demucs` (nettoyage avancé à l'import), `pip install transformers sentencepiece` (traduction).

### Docker

```bash
docker compose up -d          # GPU NVIDIA (NVIDIA Container Toolkit) ; données dans ./data
# ou : docker build -t voiceclone --build-arg TORCH=cpu --build-arg MODELS="whisper-small openvoice-v2" .
```

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
2. **Mes voix** → donnez un nom, glissez vos fichiers ou cliquez **● Enregistrer** et lisez le texte proposé (2-3 prises de 10 s). Cochez le consentement → *Créer la voix*. Vérifiez les avertissements qualité (bruit, saturation…), puis *🧬 Préparer* pour le modèle choisi.
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

### Texte tapé, appuyer pour parler, Opus

- **💬 Dire dans le Live** : tapez une phrase (ou cliquez une phrase favorite) : elle est dite par la voix clonée et mêlée au flux envoyé à Discord. Choisissez le modèle de synthèse à côté (XTTS conseillé).
- **🎙 Micro** : toujours actif, **appuyer pour parler** (touche au choix, l'onglet doit avoir le focus) ou coupé ; **Ctrl+M** bascule le micro. La **porte de bruit** n'envoie rien sous le seuil choisi.
- **Compression Opus** : le flux navigateur ↔ serveur passe de ~770 kbit/s à ~30 kbit/s (utile via un tunnel SSH) ; nécessite `pip install av` sur le serveur, sinon repli PCM automatique. Coupure réseau : reconnexion automatique (5 essais).
- **Latence estimée** et aller-retour réseau sont affichés pendant le Live.

### 🌍 Traduction vocale

En **Voix → Voix**, mode *Traduction* : choisissez la langue parlée et la langue d'arrivée. En **Live**, mode *Transcription + TTS*, champ *Traduire vers*. Il faut un modèle de traduction (Opus-MT pour fr↔en, NLLB-200 pour les autres langues) ; vers l'anglais, Whisper sait aussi traduire seul.

### 🎬 OBS et Twitch

- **Sous-titres / indicateur de parole** : *Source navigateur* OBS sur `http://localhost:7860/obs.html` (options : `?size=48&lines=1&show=subs&position=top&translation=only&key=MOTDEPASSE`).
- **Chat Twitch lu par la voix clonée** pendant le Live : `python integrations/twitch_tts.py --channel MA_CHAINE --command "!tts" --cooldown 20 --who subs` (connexion anonyme, liens masqués, délai par personne, mots interdits avec `--blocklist`). `--play --voice ID` pour jouer sur les haut-parleurs du serveur.

### Bot Discord (texte → voix dans un salon)

```bash
pip install -r discord_bot/requirements.txt
cp discord_bot/.env.example discord_bot/.env   # renseignez DISCORD_TOKEN
python discord_bot/bot.py
```

Créez l'application sur <https://discord.com/developers/applications>, invitez le bot avec les scopes `bot` + `applications.commands` et les permissions *Connect* / *Speak*. Nécessite `ffmpeg`.

| Commande | Effet |
|---|---|
| `/join`, `/leave` | rejoindre / quitter votre salon vocal |
| `/say texte [voice] [model] [language]` | lire un texte (mis en file d'attente) |
| `/mavoix voix [model] [language]` | votre voix par défaut (chaque membre la sienne) |
| `/lire on\|off` | lire à voix haute les messages de ce salon textuel (activer `DISCORD_READ_MESSAGES=1` et l'intention *Message Content*) |
| `/file`, `/skip`, `/stop` | file d'attente, passer, tout arrêter |
| `/live texte` | envoyer le texte au Live VoiceClone en cours |
| `/voices` | lister les voix |

---

## 📚 Textes longs et livres audio

Dans **Texte → Voix**, au-delà de 600 caractères (ou dès qu'il y a `[pause …]` ou des paragraphes), la génération passe en tâche de fond, phrase par phrase, avec une barre de progression. Le résultat affiche chaque phrase : écoutez-la, corrigez le texte, ↻ régénérez-la seule. Pauses : `[pause]` (0,8 s), `[pause 2s]`, `[pause 500ms]`, ligne vide = 0,6 s.

L'onglet **Livre audio** découpe un `.epub` (ordre de lecture, titres des chapitres), un `.txt` / `.md` (lignes « Chapitre … » ou « # Titre ») ou un texte collé, vous laisse relire et décocher des chapitres, puis produit un MP3 par chapitre et une archive ZIP.

## 🎛 Mélange de voix

*Mes voix → Mélanger des voix* : deux voix et un dosage (ex. 70 / 30) donnent une nouvelle voix. XTTS, OpenVoice, Chatterbox et Seed-VC mélangent réellement les empreintes ; les autres modèles utilisent une référence composée d'extraits des deux voix.

## 🎓 Entraînement

- **Audio d'entraînement** : la référence de clonage d'une voix est limitée à 30 s ; pour entraîner, ajoutez vos longs enregistrements dans *Entraînement → Audio d'entraînement* (aucune limite de durée : tout est nettoyé et découpé en phrases de 3 à 11 s, seuls les passages inutilisables sont écartés). Fichiers déjà sur le serveur : `python scripts/add_samples.py "Ma voix" interview.wav --training`.
- **Affiner XTTS v2** : à partir de l'audio d'entraînement et des échantillons transcrits de la voix (Whisper complète les transcriptions manquantes), entraîne le GPT de XTTS dans un processus séparé (progression par époque, annulation). Le modèle obtenu apparaît dans le catalogue (« XTTS · votre voix ») et devient le modèle préféré de la voix. Idéal : 5 à 30 min de voix propre ; GPU ≥ 12 Go conseillé (lot 2, accumulation 4).
- **RVC** : importez un modèle `.pth` (+ `.index`) entraîné ailleurs, ou entraînez-le depuis l'interface avec [Applio](https://github.com/IAHispano/Applio) installé à part (dossier à indiquer dans Diagnostic → Réglages). Utilisez ensuite le modèle « RVC » en Voix → Voix ou en Live.

## 🔐 Sécurité et éthique

- **Mot de passe** : lancez le serveur avec `VOICECLONE_PASSWORD=…` ; l'interface, l'API et le WebSocket sont protégés (page de connexion, ou en-tête `Authorization: Bearer …` pour les scripts et le bot).
- **Consentement** : chaque voix enregistre la déclaration faite à sa création (« c'est ma voix » / « j'ai l'autorisation »), datée.
- **Filigrane** : *Diagnostic → Réglages* ajoute un filigrane inaudible à tout ce qui est généré (résiste au MP3 et au découpage) ; *Vérifier un filigrane* dit si un fichier vient de VoiceClone (fiable à partir de ~5 s).

## 🩺 Diagnostic et mémoire GPU

L'onglet **Diagnostic** montre chaque GPU (mémoire utilisée par VoiceClone et par les autres programmes), les modèles en mémoire, les versions des bibliothèques, des vérifications courantes et le **journal du serveur** en direct (bouton *Rapport* pour demander de l'aide). Réglages : nombre maximal de modèles chargés et **déchargement automatique** du moins utilisé quand la mémoire GPU manque (les modèles d'un Live en cours ne sont jamais déchargés).

Pour tester tous les modèles installés sur le serveur : `python scripts/smoke_test.py` (chargement, synthèse, conversion, transcription, facteur temps réel ; sorties WAV à écouter dans `data/smoke_test/`).

---

## API

L'interface s'appuie sur une API REST documentée automatiquement sur <http://localhost:7860/docs>. Principaux points d'entrée :

| Méthode | Route | Description |
|---|---|---|
| GET | `/api/models` | catalogue + état (téléchargé, dépendances, chargé, progression) |
| POST | `/api/models/{id}/download` · `/load` · `/unload` | gestion des modèles |
| GET/POST | `/api/voices` | liste / création (multipart : `name`, `files[]`, `consent=true`) |
| POST | `/api/voices/{id}/samples` · `/prepare` · `/transcribe` | échantillons, préparation, transcription |
| POST | `/api/tts` | `{model_id, voice_id, text, language, params}` → WAV |
| POST | `/api/vc` | multipart `file`, `model_id`, `voice_id`, `mode` → WAV |
| GET/POST | `/api/realtime/devices` · `/start` · `/stop` · `/status` | live (audio du serveur) |
| WS | `/api/realtime/ws` | live navigateur (PCM ou Opus) |
| POST | `/api/realtime/say` | `{text}` dit par le Live en cours |
| GET | `/api/overlay` | état du Live pour OBS |
| POST | `/api/tts/long` | texte long → tâche ; `/api/jobs/{id}` pour suivre |
| POST | `/api/history/{id}/segments/{n}` | régénérer une phrase |
| POST | `/api/books/parse` · `/api/books` | livre audio |
| POST | `/api/voices/mix` | mélange de voix |
| POST | `/api/training/xtts` · `/api/training/rvc` | entraînements |
| POST | `/api/watermark/detect` | détection du filigrane |
| GET/PATCH | `/api/settings` · GET `/api/diagnostics` · `/api/logs` | réglages, diagnostic |

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
  manager.py     chargement/déchargement des moteurs, verrou GPU, LRU
  worker.py      isolation : moteur dans un autre processus / venv
  voices.py      profils de voix (nettoyage audio, référence, cache, mélange)
  prep.py        import intelligent (découpe, notation)   enhance.py  DeepFilterNet / Demucs
  realtime.py    moteur live (micro → modèle → câble virtuel)   opus.py  compression du flux
  longform.py    textes longs, pauses   books.py  livres audio   translation.py  traduction vocale
  jobs.py        file de tâches   training.py / xtts_train.py  entraînements
  watermark.py   filigrane   auth.py  mot de passe   diagnostics.py / logs.py / settings.py
  engines/       XTTS, Chatterbox, OpenVoice, F5-TTS, Seed-VC, RVC, Whisper, traduction
web/             interface (HTML/CSS/JS, sans build) + obs.html, login.html
discord_bot/     bot Discord optionnel
integrations/    lecture du chat Twitch
scripts/         install.py, smoke_test.py, add_samples.py, split_audio.py, micro virtuel Linux
tests/           tests (moteur factice, pas de téléchargement)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Les tests utilisent un moteur factice : ils vérifient l'API, la gestion des voix, les téléchargements (simulés) et les boucles temps réel sans GPU ni modèle.

## Dépannage

- **Quelque chose ne marche pas** : onglet **Diagnostic** → journal du serveur (et bouton *Rapport*), puis `python scripts/smoke_test.py --models <id>` pour tester un modèle hors interface.
- **« Dépendances manquantes »** : lancez la commande `pip install` affichée (ou `python scripts/install.py <id>`), puis redémarrez le serveur.
- **Conflit de versions entre moteurs** (numpy, transformers, torch) : `python scripts/install.py --isolated <id>`.
- **CUDA out of memory** : déchargez les modèles inutilisés (Diagnostic → *Libérer la mémoire*) ou fixez un nombre maximal de modèles chargés ; XTTS ≈ 3 Go VRAM, Chatterbox ≈ 5-6 Go, OpenVoice < 1 Go.
- **« Audio temps réel indisponible »** : installez PortAudio (`libportaudio2`).
- **Discord coupe la voix** : désactivez Krisp et la sensibilité automatique.
- **Voix robotique en live** : augmentez la taille des morceaux et le contexte, vérifiez que le seuil de silence ne coupe pas vos fins de phrases.
- **Fichier refusé** : installez `ffmpeg` pour les formats exotiques.
- **OpenVoice : `Failed to build av==10.0.0`** : le `setup.py` d'OpenVoice fige de vieilles versions (faster-whisper 0.9, av 10, numpy 1.22, gradio 3…). Installez-le **sans ses dépendances** : `pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git && pip install librosa`. VoiceClone n'utilise que son convertisseur de timbre, qui n'a besoin que de torch, librosa et soundfile.
- **`libnvrtc.so.13` / erreur `torchcodec`** : depuis PyTorch 2.9, `torchaudio` passe par `torchcodec`, qui doit être compilé pour la même version de CUDA que `torch`. VoiceClone lit l'audio de référence sans `torchcodec`, mais d'autres bibliothèques peuvent encore en avoir besoin. Pour réparer : `python -c "import torch; print(torch.__version__, torch.version.cuda)"`, puis réinstallez `torchcodec` depuis le même index que torch, par ex. `pip install --force-reinstall torchcodec --index-url https://download.pytorch.org/whl/cu128` (remplacez `cu128` par votre version de CUDA).
- **F5-TTS : « besoin du texte exact »** : F5 doit connaître le texte prononcé dans l'échantillon. Renseignez-le dans *Mes voix → Échantillons & transcription*, ou téléchargez un modèle Whisper pour qu'il soit transcrit automatiquement (une seule fois, mis en cache). F5 n'utilise qu'un échantillon de 12 s maximum : pour un long enregistrement, découpez-le aux silences avec `python scripts/split_audio.py mon_audio.wav` et importez les morceaux dans la même voix. Si le fichier est déjà sur le serveur, `python scripts/add_samples.py "Ma voix" mon_audio.wav --split` le découpe et l'ajoute directement à la voix (sans passer par le navigateur ; `--list` affiche les voix, `--create NOM --consent` en crée une).
