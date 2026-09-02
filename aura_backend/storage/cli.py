"""Fail-closed operational boundary for Phase 3 storage evidence.

The module deliberately uses only the standard library at import time.  Commands
that may later open a disposable Chroma copy import their implementation only
inside the selected handler, after explicit path and evidence validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


REQUIRED_CI_JOBS = (
    "deterministic-backend",
    "lint",
    "typing-python",
    "typing-frontend",
    "frontend-build",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_OWNERS = {"legacy", "sqlite"}
_PUBLIC_AUTHORIZATION_KEYS = {
    "authorization_id",
    "expires_at",
    "issued_at",
    "migration_plan_sha256",
    "private_summary_sha256",
    "restore_summary_sha256",
    "scope",
    "schema_version",
    "status",
}


class OperationRejected(RuntimeError):
    """A content-free operational refusal with a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"storage operation rejected: code={code}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_absolute():
        raise OperationRejected("absolute_path_required")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > max_bytes:
        raise OperationRejected("evidence_file_invalid")
    try:
        value = json.loads(resolved.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OperationRejected("evidence_json_invalid") from error
    if not isinstance(value, dict):
        raise OperationRejected("evidence_shape_invalid")
    return value


def _exclusive_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    if not path.is_absolute():
        raise OperationRejected("absolute_path_required")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def select_read_owner(marker_path: Path, *, clean_install: bool) -> str:
    """Return legacy unless a verified marker or explicit clean install selects SQLite."""
    if not marker_path.is_absolute():
        return "legacy"
    if not marker_path.exists():
        return "sqlite" if clean_install else "legacy"
    try:
        marker = _read_json(marker_path, max_bytes=16 * 1024)
    except (OperationRejected, OSError):
        return "legacy"
    owner = marker.get("owner")
    digest = marker.get("evidence_sha256")
    if (
        marker.get("schema_version") == 1
        and marker.get("status") == "pass"
        and owner in _OWNERS
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest)
    ):
        return str(owner)
    return "legacy"


def write_read_owner_marker(
    marker_path: Path,
    *,
    owner: str,
    evidence_sha256: str,
) -> dict[str, object]:
    """Atomically select one read owner from exact passing evidence."""
    if owner not in _OWNERS or not _SHA256.fullmatch(evidence_sha256):
        raise OperationRejected("read_owner_evidence_invalid")
    if not marker_path.is_absolute():
        raise OperationRejected("absolute_path_required")
    marker_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "owner": owner,
        "evidence_sha256": evidence_sha256,
    }
    staging = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.partial")
    _exclusive_json(staging, payload)
    os.replace(staging, marker_path)
    return payload


def record_authorization(
    *,
    restore_path: Path,
    migration_plan_path: Path,
    checkpoint_result: str,
    scope: str,
    private_path: Path,
    public_path: Path,
    expires_at: str,
    issued_at: str | None = None,
) -> dict[str, object]:
    """Bind exact approved evidence into exclusive private/public receipts."""
    if checkpoint_result != "approved":
        raise OperationRejected("checkpoint_not_approved")
    restore = _read_json(restore_path)
    migration = _read_json(migration_plan_path)
    if restore.get("status") != "pass" or migration.get("status") != "pass":
        raise OperationRejected("authorization_evidence_non_pass")
    if not scope or migration.get("scope") not in {None, scope}:
        raise OperationRejected("authorization_scope_mismatch")
    issued = issued_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        issued_time = datetime.fromisoformat(issued.replace("Z", "+00:00"))
        expiry_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise OperationRejected("authorization_time_invalid") from error
    if issued_time.tzinfo is None or expiry_time.tzinfo is None:
        raise OperationRejected("authorization_time_invalid")
    lifetime = (expiry_time - issued_time).total_seconds()
    if lifetime <= 0 or lifetime > 24 * 60 * 60:
        raise OperationRejected("authorization_expiry_invalid")

    restore_sha = _digest(restore_path)
    migration_sha = _digest(migration_plan_path)
    authorization_id = hashlib.sha256(
        _canonical_bytes(
            {
                "checkpoint_result": checkpoint_result,
                "expires_at": expires_at,
                "issued_at": issued,
                "migration_plan_sha256": migration_sha,
                "restore_summary_sha256": restore_sha,
                "scope": scope,
            }
        )
    ).hexdigest()
    private = {
        "schema_version": 1,
        "status": "approved",
        "authorization_id": authorization_id,
        "checkpoint_result": checkpoint_result,
        "scope": scope,
        "issued_at": issued,
        "expires_at": expires_at,
        "restore_summary_path": str(restore_path.resolve(strict=True)),
        "migration_plan_path": str(migration_plan_path.resolve(strict=True)),
        "restore_summary_sha256": restore_sha,
        "migration_plan_sha256": migration_sha,
        "restore_summary": restore,
        "migration_plan": migration,
    }
    _exclusive_json(private_path, private)
    private_sha = _digest(private_path)
    public: dict[str, object] = {
        "schema_version": 1,
        "status": "approved",
        "authorization_id": authorization_id,
        "scope": scope,
        "issued_at": issued,
        "expires_at": expires_at,
        "restore_summary_sha256": restore_sha,
        "migration_plan_sha256": migration_sha,
        "private_summary_sha256": private_sha,
    }
    if set(public) != _PUBLIC_AUTHORIZATION_KEYS:
        raise OperationRejected("authorization_public_shape_invalid")
    _exclusive_json(public_path, public)
    return public


def evaluate_ci_evidence(
    payload: Mapping[str, Any],
    *,
    expected_sha: str,
    expected_workflow_digest: str,
) -> dict[str, object]:
    """Require the exact current revision and all deterministic jobs green."""
    reasons: list[str] = []
    if not _GIT_SHA.fullmatch(expected_sha) or payload.get("head_sha") != expected_sha:
        reasons.append("head_sha_mismatch")
    if (
        not _SHA256.fullmatch(expected_workflow_digest)
        or payload.get("workflow_digest") != expected_workflow_digest
    ):
        reasons.append("workflow_digest_mismatch")
    jobs_value = payload.get("jobs")
    jobs = jobs_value if isinstance(jobs_value, list) else []
    by_name = {
        str(job.get("name")): job
        for job in jobs
        if isinstance(job, Mapping) and isinstance(job.get("name"), str)
    }
    results: dict[str, str] = {}
    for name in REQUIRED_CI_JOBS:
        job = by_name.get(name)
        if (
            job is None
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            reasons.append(f"required_job_non_pass:{name}")
            results[name] = "non_pass"
        else:
            results[name] = "pass"
    return {
        "status": "pass" if not reasons else "non_pass",
        "required_jobs": results,
        "reason_codes": reasons,
    }


def prepare_ci_publication(
    *,
    repository_root: Path,
    workflow_path: Path,
    output_path: Path,
    expires_at: str,
) -> dict[str, object]:
    """Create a local-only proposal bound to a clean current implementation SHA."""
    root = repository_root.resolve(strict=True)
    if not workflow_path.resolve(strict=True).is_relative_to(root):
        raise OperationRejected("workflow_outside_repository")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if status.returncode or status.stdout or head.returncode:
        raise OperationRejected("worktree_not_clean")
    implementation_sha = head.stdout.strip()
    if not _GIT_SHA.fullmatch(implementation_sha):
        raise OperationRejected("implementation_sha_invalid")
    proposal: dict[str, object] = {
        "schema_version": 1,
        "status": "awaiting_human_action",
        "remote": "origin",
        "preferred_ref": "refs/heads/phase-03-memory-integrity-ci",
        "optional_ref": "refs/heads/main",
        "implementation_sha": implementation_sha,
        "workflow_path": str(workflow_path.resolve(strict=True).relative_to(root)),
        "workflow_digest": _digest(workflow_path.resolve(strict=True)),
        "allowed_actions": [
            "non_force_branch_push",
            "workflow_dispatch_existing_exact_ref",
        ],
        "expires_at": expires_at,
    }
    _exclusive_json(output_path, proposal)
    return proposal


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed Phase 3 command grammar without importing integrations."""
    parser = argparse.ArgumentParser(prog="python -m aura_backend.storage.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    def path_option(command: argparse.ArgumentParser, name: str) -> None:
        command.add_argument(f"--{name}", required=True, type=Path)

    for name in ("benchmark", "snapshot-verify", "migrate-authorized-copy"):
        command = commands.add_parser(name)
        path_option(command, "evidence")
        path_option(command, "output")

    authorization = commands.add_parser("record-authorization")
    for name in ("restore", "migration-plan", "private", "public"):
        path_option(authorization, name)
    authorization.add_argument("--checkpoint-result", required=True)
    authorization.add_argument("--scope", required=True)
    authorization.add_argument("--expires-at", required=True)

    for name in ("switch-reads", "rollback-reads"):
        command = commands.add_parser(name)
        path_option(command, "marker")
        path_option(command, "evidence")

    publication = commands.add_parser("prepare-ci-publication")
    for name in ("repository-root", "workflow", "output"):
        path_option(publication, name)
    publication.add_argument("--expires-at", required=True)

    for name in ("verify-ci-publication", "collect-ci-evidence", "validate-evidence"):
        command = commands.add_parser(name)
        path_option(command, "evidence")
        path_option(command, "output")
        command.add_argument("--expected-sha", required=True)
        command.add_argument("--workflow-digest", required=True)
    return parser


def _dispatch(arguments: argparse.Namespace) -> dict[str, object]:
    command = str(arguments.command)
    if command == "record-authorization":
        return record_authorization(
            restore_path=arguments.restore,
            migration_plan_path=arguments.migration_plan,
            checkpoint_result=arguments.checkpoint_result,
            scope=arguments.scope,
            private_path=arguments.private,
            public_path=arguments.public,
            expires_at=arguments.expires_at,
        )
    if command in {"switch-reads", "rollback-reads"}:
        evidence = _read_json(arguments.evidence)
        digest = _digest(arguments.evidence)
        if evidence.get("status") != "pass":
            raise OperationRejected("read_owner_evidence_non_pass")
        owner = "sqlite" if command == "switch-reads" else "legacy"
        return write_read_owner_marker(
            arguments.marker,
            owner=owner,
            evidence_sha256=digest,
        )
    if command == "prepare-ci-publication":
        return prepare_ci_publication(
            repository_root=arguments.repository_root,
            workflow_path=arguments.workflow,
            output_path=arguments.output,
            expires_at=arguments.expires_at,
        )
    if command in {"verify-ci-publication", "collect-ci-evidence", "validate-evidence"}:
        evidence = _read_json(arguments.evidence)
        result = evaluate_ci_evidence(
            evidence,
            expected_sha=arguments.expected_sha,
            expected_workflow_digest=arguments.workflow_digest,
        )
        _exclusive_json(arguments.output, result)
        return result
    # Real benchmark/snapshot/import work needs later human-gated evidence and
    # remains unavailable from this preparatory plan rather than guessing paths.
    _read_json(arguments.evidence)
    raise OperationRejected("later_checkpoint_required")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
) -> int:
    """Run one command with fixed content-free JSON and stable exit classes."""
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        result = _dispatch(arguments)
    except (OperationRejected, FileExistsError, FileNotFoundError, OSError) as error:
        code = error.code if isinstance(error, OperationRejected) else "operation_failed"
        stdout.write(json.dumps({"status": "non_pass", "code": code}) + "\n")
        return 4
    stdout.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
    return 0 if result.get("status") in {"pass", "approved", "awaiting_human_action"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
