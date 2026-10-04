# Plan de travail — améliorations VoiceClone

Légende : effort **S** (quelques heures) · **M** (une journée) · **L** (plusieurs jours).
État : ☐ à faire · ◐ en cours · ☑ fait (testé ici) · ⚠ fait, non testé sur vrai modèle/GPU (à valider sur le serveur).

> Contexte de test : l'environnement de développement n'a ni GPU ni PyTorch ni accès à Hugging Face.
> Tout ce qui ne dépend pas d'un vrai modèle est couvert par des tests automatiques (`pytest`) et
> vérifié dans Chromium. Ce qui dépend d'un vrai modèle est codé d'après le code source des
> bibliothèques et marqué ⚠.

## Lot 1 — Interface et confort (S)
| # | Tâche | Effort | État |
|---|---|---|---|
| 1.1 | Renommer « Entraîner » → « Préparer » | S | ☑ |
| 1.2 | Choisir l'échantillon de référence : réordonner, épingler le meilleur en premier | S | ☑ |
| 1.3 | Comparateur A/B : même texte, plusieurs modèles côte à côte | S | ☑ |
| 1.4 | Historique : favoris, filtres, « régénérer avec les mêmes réglages » | S | ☑ |
| 1.5 | Réglages par voix (modèle et paramètres préférés mémorisés) | S | ☑ |

## Lot 2 — Live / Discord (S-M)
| # | Tâche | Effort | État |
|---|---|---|---|
| 2.1 | Texte → Discord dans le Live + phrases favorites | S | ☑ |
| 2.2 | Appuyer pour parler / raccourci muet + porte de bruit navigateur | S | ☑ |
| 2.3 | Reconnexion automatique du Live | S | ☑ |
| 2.4 | Latence : recouvrement + alignement des morceaux, préchauffage, affichage de la latence réelle | M | ⚠ |
| 2.5 | Compression Opus du flux (WebCodecs ↔ PyAV), repli PCM automatique | M | ☑ |

## Lot 3 — Robustesse et exploitation (S-M)
| # | Tâche | Effort | État |
|---|---|---|---|
| 3.1 | Script de test sur vrais modèles (`scripts/smoke_test.py`) | S | ☑ |
| 3.2 | Mémoire GPU : affichage VRAM + déchargement automatique du moins utilisé | S | ⚠ |
| 3.3 | Page Diagnostic : versions installées, journaux du serveur dans l'interface | S | ☑ |
| 3.4 | File de tâches (progression, annulation) pour les longues générations | M | ☑ |
| 3.5 | Scripts d'installation par modèle + Dockerfile | M | ⚠ |

## Lot 4 — Sécurité et éthique (S-M)
| # | Tâche | Effort | État |
|---|---|---|---|
| 4.1 | Mot de passe sur l'interface et l'API (`VOICECLONE_PASSWORD`) | S | ☑ |
| 4.2 | Trace du consentement enregistrée avec chaque voix | S | ☑ |
| 4.3 | Filigrane uniforme optionnel sur toutes les sorties + outil de détection | M | ☑ |

## Lot 5 — Audio et textes longs (M)
| # | Tâche | Effort | État |
|---|---|---|---|
| 5.1 | Forme d'onde avec sélection avant import | M | ☑ |
| 5.2 | Textes longs : phrase par phrase, régénérer une phrase, pauses `[pause 1s]`, export MP3 | M | ☑ |
| 5.3 | Nettoyage avancé à l'import : DeepFilterNet et Demucs (optionnels, repli automatique) | M | ⚠ |

## Lot 6 — Nouvelles fonctions (M)
| # | Tâche | Effort | État |
|---|---|---|---|
| 6.1 | Traduction vocale (tu parles français → ça sort en anglais avec ta voix) | M | ⚠ |
| 6.2 | Mode livre audio (texte / .txt / .epub → chapitres audio, archive ZIP) | M | ☑ |
| 6.3 | Mélange de voix (voix intermédiaire pondérée entre deux profils) | M | ⚠ |
| 6.4 | Intégrations : page OBS, lecture du chat Twitch, bot Discord (file, voix par utilisateur) | M | ⚠ |

## Lot 7 — Gros chantiers modèles (L)
| # | Tâche | Effort | État |
|---|---|---|---|
| 7.1 | Seed-VC : conversion zero-shot temps réel de meilleure qualité | M | ⚠ |
| 7.2 | RVC : inférence de modèles .pth/.index + entraînement via Applio | L | ⚠ |
| 7.3 | Fine-tuning XTTS (jeu de données automatique, entraînement en tâche de fond, modèle fine-tuné sélectionnable) | L | ⚠ |
| 7.4 | Isolation des moteurs : un processus (et un Python/venv au choix) par modèle | L | ⚠ |

## À valider sur le serveur (éléments ⚠)

Tout le reste est couvert par `pytest` (78 tests) et vérifié dans Chromium avec un moteur factice.
Commencez par `git pull`, relancez le serveur, puis **Ctrl+F5** dans le navigateur.

| # | Comment vérifier | En cas de problème |
|---|---|---|
| 2.4 | Live en conversion directe : la voix doit être continue, sans « clics » aux jointures. La latence estimée s'affiche. | Augmenter « Fondu » / « Contexte » dans les réglages avancés |
| 3.2 | Diagnostic : la VRAM de chaque GPU et la mémoire de chaque modèle chargé s'affichent ; avec « Nombre maximal » = 1, charger un 2e modèle décharge le 1er. | Journal du serveur (Diagnostic) |
| 3.5 | `python scripts/install.py --list` ; `docker compose build` (non testé ici : réseau du bac à sable bloqué). | |
| 5.3 | `pip install deepfilternet` puis import intelligent avec « DeepFilterNet » ; `pip install demucs` pour « Demucs ». Le rapport indique la méthode utilisée ou le repli. | Repli spectral automatique |
| 6.1 | Télécharger « Opus-MT fr→en » (`pip install transformers sentencepiece`), Voix → Voix, mode Traduction. | |
| 6.3 | Mélanger deux voix puis générer avec XTTS / OpenVoice / Chatterbox. | Supprimer le cache : bouton Préparer |
| 6.4 | Ouvrir `obs.html` pendant un Live ; `python integrations/twitch_tts.py --channel <chaîne>` ; bot Discord : `/mavoix`, `/file`, `/lire`. | |
| 7.1 | Seed-VC : `python scripts/install.py --isolated seed-vc`, télécharger, tester en Voix → Voix puis en Live (4-8 étapes). | `python scripts/smoke_test.py --models seed-vc` |
| 7.2 | RVC : `python scripts/install.py --isolated rvc`, télécharger, importer un .pth sur une voix (onglet Entraînement), convertir. Applio : renseigner son dossier dans Diagnostic. | |
| 7.3 | Affiner XTTS (2-10 min d'audio transcrit, 10 époques) : un modèle « XTTS · voix » apparaît dans Texte → Voix. | Journal + `data/training/*/train.log` |
| 7.4 | Diagnostic → Réglages → Isolation : renseigner un Python ; le modèle se charge dans un processus séparé. | |
