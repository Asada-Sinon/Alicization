"""Launch the live dashboard: starts the sim + web server and opens a browser.

Usage:
    .venv/bin/python scripts/run_live.py [--host 0.0.0.0] [--port 8000] [--no-open]
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

sys.path.insert(0, ".")

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not args.no_open:
        url = f"http://localhost:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # Import here so the sim (and JAX) initialize inside the server process.
    from server.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
