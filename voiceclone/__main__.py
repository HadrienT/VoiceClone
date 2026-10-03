"""Lancement : python -m voiceclone [--host 127.0.0.1] [--port 7860]"""

from __future__ import annotations

import argparse
import logging
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceclone", description="Studio de clonage de voix local")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (0.0.0.0 pour le réseau local)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir le navigateur")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    from .server import create_app

    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"
    print(f"\n  VoiceClone est prêt : {url}\n")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
