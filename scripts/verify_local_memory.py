"""Live Aura memory/maintenance check using an isolated ledger and synthetic facts.

Run with installed Ollama models and the Memvid extra. Leaves a compact report
in --output; the temporary server logs/ledger are removed after owned shutdown.
This small smoke test is evidence of these cases, not a general recall benchmark.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/aura-live-memory-report.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report: dict = {"status": "RUNNING", "checks": {}}

    def record(name: str, passed: bool) -> None:
        report["checks"][name] = passed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            raise AssertionError(name)

    try:
        with tempfile.TemporaryDirectory(prefix="aura-live-memory-") as folder:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            environment = dict(
                os.environ, AURA_LEDGER_DIRECTORY=folder, AURA_CLEAN_INSTALL="true",
                AURA_MEMVID_ENABLED="true", AUTONOMIC_ENABLED="true",
                AURA_RELOAD="false", MEMVID_TELEMETRY="0",
            )
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base, timeout=300) as client:
                def post(path: str, body: dict) -> dict:
                    result = client.post(path, json=body)
                    result.raise_for_status()
                    return result.json()

                def chat(message: str, key: str, session: str) -> dict:
                    return post("/conversation", {
                        "user_id": "synthetic-check", "session_id": session,
                        "message": message, "idempotency_key": key,
                    })

                for iteration in range(2):
                    print(f"Starting isolated Aura instance {iteration + 1}", flush=True)
                    with open(Path(folder) / f"server-{iteration}.log", "w+") as log:
                        process = subprocess.Popen(
                            ["sh", "./start_full_system.sh", "--backend-only", "--port", str(port)],
                            cwd=root, env=environment, stdout=log, stderr=log,
                        )
                        try:
                            for _ in range(120):
                                if process.poll() is not None:
                                    raise RuntimeError("Owned smoke server exited before readiness")
                                try:
                                    if client.get("/ready", timeout=2).status_code == 200:
                                        break
                                except httpx.HTTPError:
                                    pass
                                time.sleep(1)
                            else:
                                raise TimeoutError("Owned smoke server did not become ready")

                            for _ in range(60):
                                status = client.get("/autonomic/status").json().get("system_status", {})
                                if (status.get("last_memory_maintenance") or {}).get("status") == "completed":
                                    break
                                time.sleep(1)
                            record(f"maintenance_after_start_{iteration}",
                                   (status.get("last_memory_maintenance") or {}).get("status") == "completed")
                            if iteration == 0:
                                first = chat("Our test vault code is AMBER-7429. Acknowledge briefly.", "first", "continuity")
                                record("durable_commit", first["emotional_state"]["simulation"]["disposition"] == "committed")
                                second = chat("What is our vault code? Reply with the code only.", "second", "continuity")
                                record("same_session_recall", "AMBER-7429" in second["response"])
                                cross = chat("Which access code did I give you for the test vault? Reply briefly.", "cross", "new-session")
                                record("cross_session_recall", "AMBER-7429" in cross["response"])
                                correction = chat("Correction: our test vault code is now VIOLET-8631, replacing AMBER-7429. Acknowledge briefly.", "correction", "continuity")
                                record("correction_committed", correction["emotional_state"]["simulation"]["disposition"] == "committed")
                                archive = post("/memvid/archive-session", {"user_id": "synthetic-check", "session_id": "continuity"})
                                record("copy_only_archive", archive["messages_archived"] == 6)
                            else:
                                recall = chat("What is the current test vault code? Give only the current code.", "restart", "restart-session")
                                record("corrected_recall_after_restart", "VIOLET-8631" in recall["response"] and "AMBER-7429" not in recall["response"])
                                absent = chat("What was my childhood dog's name? If I haven't told you, reply only: I don't know.", "absent", "absence")
                                record("absent_fact_abstention", "don't know" in absent["response"].lower().replace("’", "'"))
                            found = post("/search", {"user_id": "synthetic-check", "query": "vault", "include_active": True, "include_archives": False})
                            record(f"short_query_search_{iteration}", any("VIOLET-8631" in item["content"] for item in found["results"]))
                            other = post("/search", {"user_id": "other", "query": "vault", "include_active": True, "include_archives": False})
                            record(f"cross_user_isolation_{iteration}", not other["results"])
                            archive_status = client.get("/memvid/status?user_id=synthetic-check").json()
                            record(f"archive_after_start_{iteration}", archive_status["status"] == "operational" and archive_status["archives_count"] == 1)
                            found = post("/search", {"user_id": "synthetic-check", "query": "vault access code", "include_active": False, "include_archives": True})
                            record(f"archive_recall_{iteration}", any("VIOLET-8631" in item["content"] for item in found["results"]))
                        finally:
                            process.send_signal(signal.SIGINT)
                            try:
                                process.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                process.terminate()
                                process.wait(timeout=15)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error_type"] = type(error).__name__
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
