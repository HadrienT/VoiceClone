"""Vérifications statiques : un nom utilisé mais jamais importé ne se voit qu'au chargement du moteur
(sur le serveur GPU, pas dans les tests avec le moteur factice). pyflakes le détecte sans rien exécuter."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_no_undefined_names():
    pytest.importorskip("pyflakes")
    r = subprocess.run([sys.executable, "-m", "pyflakes", "voiceclone", "scripts", "integrations", "discord_bot"],
                       capture_output=True, text=True, cwd=ROOT)
    serious = [line for line in r.stdout.splitlines()
               if "undefined name" in line or "syntax" in line.lower() or "referenced before assignment" in line]
    assert not serious, "\n".join(serious)


def test_pkg_resources_shim():
    code = ("import sys; sys.modules['pkg_resources'] = None\n"  # simule setuptools >= 81
            "from voiceclone.engines.base import ensure_pkg_resources\nensure_pkg_resources()\n"
            "import pkg_resources\n"
            "assert pkg_resources.get_distribution('numpy').version\n"
            "try:\n    pkg_resources.get_distribution('paquet-inexistant-xyz')\n"
            "except pkg_resources.DistributionNotFound:\n    pass\nelse:\n    raise SystemExit('pas d erreur')\n"
            "import os, scipy; assert pkg_resources.resource_filename('scipy', 'x').endswith(os.path.join('scipy', 'x'))\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
