"""Run the local backend independently of an editor terminal and reloader."""
import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from threading import Event, Thread

import uvicorn


def main():
    # Redirected Windows pipes otherwise inherit cp1252; an emoji/progress
    # message from data providers must never turn a successful fetch into an error.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    state_file = Path(args.state_file).resolve()
    runtime_dir = (root / "outputs" / "backend-runtime").resolve()
    if state_file.parent != runtime_dir:
        raise ValueError("State file must stay inside this project's backend-runtime directory")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stop_file = runtime_dir / f"stop-{os.getpid()}.flag"
    server = uvicorn.Server(uvicorn.Config("src.backend.app:app", host="127.0.0.1", port=8000,
        reload=False, timeout_graceful_shutdown=5))
    finished = Event()

    def watch_stop_request():
        while not finished.wait(.2):
            if stop_file.exists():
                server.should_exit = True
                from src.backend.services.portfolio_progress import cancel_active_portfolio_analyses
                cancel_active_portfolio_analyses()
                return

    temporary_state = state_file.with_name(f"state-{os.getpid()}.tmp")
    temporary_state.write_text(json.dumps({"pid": os.getpid(), "project_root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(), "stop_file": str(stop_file),
        "host": "127.0.0.1", "port": 8000, "reload": False}), encoding="utf-8")
    temporary_state.replace(state_file)
    Thread(target=watch_stop_request, name="backend-stop-request", daemon=True).start()
    try:
        server.run()
    finally:
        finished.set()
        if state_file.exists():
            try:
                if json.loads(state_file.read_text(encoding="utf-8")).get("pid") == os.getpid():
                    state_file.unlink()
            except (OSError, ValueError):
                pass
        stop_file.unlink(missing_ok=True)
        temporary_state.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
