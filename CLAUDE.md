# Consignes pour Claude Code

- **Fusion continue autorisée** : le propriétaire du dépôt fait `git pull` sur `main`. Une fois un travail
  testé (`pytest` vert), poussez la branche de travail, ouvrez une PR vers `main` et fusionnez-la sans
  attendre de confirmation.
- Interface et messages en français.
- Avant de pousser : `pytest`, `node --check web/*.js`, et une vérification dans Chromium pour les
  changements d'interface (moteur factice : `tests/fake_engine.py`).
- Le suivi des améliorations est dans `WORKPLAN.md` (☑ testé, ⚠ à valider sur le serveur GPU).
