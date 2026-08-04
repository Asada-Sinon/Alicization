"""Launch the live dashboard: starts the sim + web server and opens a browser.

Usage:
    .venv/bin/python scripts/run_live.py [--host 0.0.0.0] [--port 8000] [--no-open]
                                         [--set FIELD=VALUE ...]

`--set` takes the same ablation arms as `run_headless.py`, so a config can be
*looked at* and not only measured -- see `server.app._config_from_env`.
"""

from __future__ import annotations

import argparse
import os
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
    ap.add_argument("--speed", type=int, default=None, metavar="STEPS",
                    help="initial sim steps per rendered frame (the FLA dial, 1..2000). "
                         "The dashboard slider can change it live, but a headless "
                         "verification run has no hands -- and at the default 4 steps "
                         "per frame the interesting arms need minutes of wall clock "
                         "before anything has evolved to look at.")
    ap.add_argument("--set", action="append", metavar="FIELD=VALUE", dest="sets",
                    help="override a Config field, e.g. --set fruit_energy=4.0. "
                         "Repeatable. Same arms as run_headless.py")
    args = ap.parse_args()

    # The sim is built at `server.app` import time, so the arm has to be in place
    # before that import below -- hence an env var rather than a function argument.
    if args.sets:
        os.environ["UNDERWORLD_SET"] = ";".join(args.sets)
    if args.speed is not None:
        os.environ["UNDERWORLD_SPEED"] = str(args.speed)

    if not args.no_open:
        url = f"http://localhost:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # Import here so the sim (and JAX) initialize inside the server process.
    from server.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
