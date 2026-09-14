"""Opt-in live smoke: real launcher, model, embeddings, archive, and restart.

Uses an isolated temporary ledger and synthetic messages. Never prints model
reasoning, credentials, or personal memories. Run with the installed memvid extra.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report: dict = {}
    with tempfile.TemporaryDirectory(prefix="aura-live-memory-") as folder:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        environment = dict(os.environ, AURA_LEDGER_DIRECTORY=folder,
                           AURA_CLEAN_INSTALL="true", AURA_MEMVID_ENABLED="true",
                           AUTONOMIC_ENABLED="false", AURA_RELOAD="false", MEMVID_TELEMETRY="0")
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, timeout=240) as client:
            for iteration in range(2):
                with open(Path(folder) / f"server-{iteration}.log", "w+") as log:
                    process = subprocess.Popen(
                        ["sh", "./start_full_system.sh", "--backend-only", "--port", str(port)],
                        cwd=root, env=environment, stdout=log, stderr=log,
                    )
                    try:
                        for _ in range(120):
                            if process.poll() is not None:
                                log.seek(0)
                                raise RuntimeError("Owned smoke server exited before readiness: " + log.read()[-4000:])
                            try:
                                if client.get("/ready", timeout=2).status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(1)
                        else:
                            raise TimeoutError("Owned smoke server did not become ready")

                        def post(path: str, body: dict) -> dict:
                            result = client.post(path, json=body)
                            result.raise_for_status()
                            return result.json()

                        if iteration == 0:
                            common = {"user_id": "synthetic-check", "session_id": "continuity"}
                            first = post("/conversation", {**common, "message": "Our test vault code is AMBER-7429. Acknowledge briefly.", "idempotency_key": "first"})
                            assert first["emotional_state"]["simulation"]["disposition"] == "committed", first
                            second = post("/conversation", {**common, "message": "What is our vault code? Reply with the code only.", "idempotency_key": "second"})
                            assert "AMBER-7429" in second["response"], second["response"]
                            report["continuity"] = second["response"]
                            report["archive"] = post("/memvid/archive-session", common)
                            assert report["archive"]["messages_archived"] == 4
                        status = client.get("/memvid/status?user_id=synthetic-check").json()
                        assert status["status"] == "operational" and status["archives_count"] == 1, status
                        found = post("/search", {"user_id": "synthetic-check", "query": "vault access code", "include_active": False, "include_archives": True})
                        assert any("AMBER-7429" in item["content"] for item in found["results"]), found
                        report[f"archive_search_after_start_{iteration}"] = len(found["results"])
                        assert client.get("/memvid/status?user_id=other").json()["archives_count"] == 0
                    finally:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            process.wait(timeout=10)
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
