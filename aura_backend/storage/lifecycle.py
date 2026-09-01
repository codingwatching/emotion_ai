"""Truthful snapshot, restore, export, and deletion ownership for Aura storage.

Every filesystem or destructive target is explicit, contained, and suitable for
synthetic or separately authorized roots. This module never discovers or opens
historical Aura stores, backups, or archives on its own.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from aura_backend.storage.connection import open_database
from aura_backend.storage.models import StorageFailure
from aura_backend.storage.projection import ProjectionAdapter
from aura_backend.storage.repository import StorageRepository
from aura_backend.storage.schema import rebuild_fts
from aura_backend.runtime_security import (
    StoragePathError,
    safe_export_format,
    safe_storage_component,
)


class SnapshotStatus(str, Enum):
    """Publication state returned only for a complete database/manifest pair."""

    PUBLISHED = "published"


class RestoreStatus(str, Enum):
    """Restore state returned only after the complete required gate passes."""

    COMPLETE = "complete"


class ExportStatus(str, Enum):
    """Publication state for a complete versioned scope export."""

    PUBLISHED = "published"


class DeletionAction(str, Enum):
    """Only destructive action names accepted by the lifecycle boundary."""

    SESSION = "session"
    SCOPE = "scope"
    GENERATED_EXPORT = "generated_export"
    PROJECTION_GENERATION = "projection_generation"
    ARCHIVE = "archive"
    BACKUP_GENERATION = "backup_generation"


class DeletionStatus(str, Enum):
    """Truthful final state after exact post-execution verification."""

    BLOCKED = "blocked"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class ProjectionFactory(Protocol):
    """Construct a disposable projection rooted inside one isolated restore."""

    def __call__(
        self, root: Path, repository: StorageRepository
    ) -> ProjectionAdapter: ...


class FixtureVerifier(Protocol):
    """Run the fixed retrieval fixture classes against an isolated restore."""

    def __call__(
        self, repository: StorageRepository, adapter: ProjectionAdapter
    ) -> Mapping[str, bool]: ...


SnapshotFaultHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Content-free identity for an atomically published SQLite snapshot."""

    status: SnapshotStatus
    snapshot_id: str
    database_path: Path
    manifest_path: Path
    database_sha256: str
    manifest_sha256: str
    database_size: int


@dataclass(frozen=True, slots=True)
class RestoreResult:
    """Content-free evidence for one complete isolated restore."""

    status: RestoreStatus
    database_path: Path
    checks: tuple[str, ...]
    table_counts: dict[str, int]
    manifest_table_counts: dict[str, int]
    table_digests: dict[str, str]
    manifest_table_digests: dict[str, str]
    projection_origin_count: int
    projection_ids_sha256: str


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Content-free identity and parity facts for one published JSON export."""

    status: ExportStatus
    path: Path
    manifest_sha256: str
    counts: dict[str, int]
    record_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class ExportVerification:
    """Round-trip comparison between one export and current scoped truth."""

    valid: bool
    counts: dict[str, int]
    record_hashes: dict[str, str]


DeletionInventory = Callable[[DeletionAction, str, str | None], Mapping[str, str]]
DeletionCallback = Callable[[tuple[str, ...]], None]
DeletionRemaining = Callable[[tuple[str, ...]], Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class DeletionTarget:
    """Injected active target with exact content-free inventory operations."""

    name: str
    kind: str
    inventory: DeletionInventory
    delete: DeletionCallback
    remaining: DeletionRemaining
    reconcile: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class ResidualCopy:
    """Content-free retained copy which normal active deletion never opens."""

    kind: str
    alias: str


@dataclass(frozen=True, slots=True)
class RestoreProof:
    """Independent exact restore evidence required for retained-copy purge."""

    generation_id: str
    manifest_sha256: str
    checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeletionItem:
    """One exact approved canonical, FTS, or injected active-target identity."""

    target: str
    identity: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class DeletionPlan:
    """Expiring HMAC-bound inventory; possessing it is not confirmation."""

    plan_id: str
    action: DeletionAction
    scope_id: str
    target_id: str | None
    policy_version: int
    created_at: float
    expires_at: float
    inventory_digest: str
    counts: dict[str, int]
    items: tuple[DeletionItem, ...]
    retained_copies: tuple[ResidualCopy, ...]
    restore_proof_generation: str | None
    challenge: str


@dataclass(frozen=True, slots=True)
class DeletionConfirmation:
    """Short-lived single-use confirmation bound to one immutable plan."""

    confirmation_id: str
    plan_digest: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class DeletionExecution:
    """Content-free execution receipt awaiting exact verification."""

    execution_id: str
    plan_digest: str
    canonical_committed: bool
    attempted_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeletionResult:
    """Verified application deletion truth without forensic-erasure claims."""

    status: DeletionStatus
    action: DeletionAction
    scope_id: str
    inventory_digest: str
    deleted_counts: dict[str, int]
    failed_targets: tuple[str, ...]
    retry_targets: tuple[str, ...]
    retained_copies: tuple[ResidualCopy, ...]
    application_level_deletion: bool
    physical_purge: str
    forensic_erasure: bool


_SAFE_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHUNK_BYTES = 1024 * 1024
_SNAPSHOT_FORMAT_VERSION = 1
_SNAPSHOT_CONFIG_VERSION = 1
_FTS_CONFIG = "unicode61 remove_diacritics 2"
_CANONICAL_TABLES = (
    "memory_scopes",
    "sessions",
    "turns",
    "events",
    "derived_memories",
    "memory_sources",
    "memory_supersessions",
    "memory_retractions",
    "profile_versions",
    "legacy_sources",
    "projection_generations",
    "retrieval_runs",
    "retrieval_candidates",
    "legacy_fragments",
    "legacy_import_evidence",
)
_FTS_TABLES = ("event_fts", "memory_fts")
_EXPORT_RECORDS = (
    "sessions",
    "turns",
    "events",
    "derived_memories",
    "memory_sources",
    "memory_supersessions",
    "memory_retractions",
    "profile_versions",
)
_PRIVATE_PROFILE_KEYS = {
    "api_key",
    "apikey",
    "credential",
    "credentials",
    "cursor_secret",
    "embedding",
    "embeddings",
    "internal_path",
    "password",
    "provider_api_key",
    "secret",
    "token",
}
_DELETION_POLICY_VERSION = 1
_DELETION_MAX_TTL_SECONDS = 15 * 60
_ACTIVE_DELETION_KINDS = {"projection", "cache", "generated_export"}
_TARGET_ACTION_KIND = {
    DeletionAction.GENERATED_EXPORT: "generated_export",
    DeletionAction.PROJECTION_GENERATION: "projection_generation",
    DeletionAction.ARCHIVE: "archive",
    DeletionAction.BACKUP_GENERATION: "backup_generation",
}
_CANONICAL_DELETE_ORDER = (
    "retrieval_candidates",
    "retrieval_runs",
    "memory_supersessions",
    "memory_retractions",
    "memory_sources",
    "legacy_sources",
    "derived_memories",
    "events",
    "turns",
    "sessions",
    "profile_versions",
    "legacy_fragments",
    "memory_scopes",
)
_DELETION_PRIMARY_KEYS = {
    "memory_scopes": ("scope_id",),
    "sessions": ("session_id",),
    "turns": ("turn_id",),
    "events": ("event_id",),
    "derived_memories": ("memory_id",),
    "memory_sources": ("memory_id", "event_id"),
    "memory_supersessions": ("old_memory_id", "new_memory_id"),
    "memory_retractions": ("memory_id",),
    "profile_versions": ("scope_id", "profile_version"),
    "legacy_sources": ("root_fingerprint", "collection_name", "legacy_id"),
    "retrieval_runs": ("run_id",),
    "retrieval_candidates": ("run_id", "origin_id"),
    "legacy_fragments": ("fragment_id",),
}


class LifecycleService:
    """Single injected owner for recoverable SQLite lifecycle operations."""

    FIXTURE_CHECKS = (
        "direct_retrieval",
        "correction_freshness",
        "provenance",
        "pagination",
        "cross_scope_isolation",
    )
    REQUIRED_RESTORE_CHECKS = (
        "manifest_hash",
        "database_hash",
        "integrity_check",
        "foreign_key_check",
        "schema_parity",
        "table_count_parity",
        "table_digest_parity",
        "fts_rebuild",
        "projection_parity",
        *FIXTURE_CHECKS,
        "active_source_invariance",
    )

    def __init__(
        self,
        *,
        repository: StorageRepository,
        projection_factory: ProjectionFactory,
        fixture_verifier: FixtureVerifier,
        tool_commit: str,
        deletion_secret: bytes | None = None,
        deletion_clock: Callable[[], float] | None = None,
        deletion_targets: tuple[DeletionTarget, ...] = (),
        retained_copies: tuple[ResidualCopy, ...] = (),
    ) -> None:
        if not repository.database_path.is_absolute():
            raise StorageFailure("absolute_path_required")
        if not tool_commit:
            raise StorageFailure("lifecycle_tool_commit_required")
        self.repository = repository
        self.projection_factory = projection_factory
        self.fixture_verifier = fixture_verifier
        self.tool_commit = tool_commit
        secret = deletion_secret if deletion_secret is not None else os.urandom(32)
        if len(secret) < 16:
            raise StorageFailure("deletion_secret_too_short")
        names = tuple(target.name for target in deletion_targets)
        if len(names) != len(set(names)) or any(
            not _SAFE_OPERATION_ID.fullmatch(name) for name in names
        ):
            raise StorageFailure("deletion_target_invalid")
        if any(
            target.kind
            not in {
                *_ACTIVE_DELETION_KINDS,
                "projection_generation",
                "archive",
                "backup_generation",
            }
            for target in deletion_targets
        ):
            raise StorageFailure("deletion_target_kind_invalid")
        if any(
            not copy.kind
            or not copy.alias
            or len(copy.kind) > 128
            or len(copy.alias) > 512
            for copy in retained_copies
        ):
            raise StorageFailure("deletion_residual_invalid")
        self._deletion_secret = secret
        self._deletion_clock = deletion_clock or time.monotonic
        self._deletion_targets = tuple(
            sorted(deletion_targets, key=lambda target: target.name)
        )
        self._retained_copies = tuple(
            sorted(retained_copies, key=lambda copy: (copy.kind, copy.alias))
        )
        self._confirmed_deletions: dict[str, str] = {}
        self._consumed_confirmations: set[str] = set()

    def create_snapshot(
        self,
        destination_root: Path,
        *,
        snapshot_id: str,
        timeout_seconds: float = 60.0,
        max_database_bytes: int = 64 * 1024**4,
        pages_per_step: int = 128,
        busy_sleep_seconds: float = 0.01,
        fault_hook: SnapshotFaultHook | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> SnapshotResult:
        """Publish one bounded online SQLite snapshot plus canonical manifest."""
        source, root = self._validate_snapshot_paths(destination_root, snapshot_id)
        if timeout_seconds <= 0 or not 1 <= pages_per_step <= 65_536:
            raise StorageFailure("snapshot_bounds_invalid", identifier=snapshot_id)
        source_size = source.stat().st_size
        if max_database_bytes <= 0 or source_size > max_database_bytes:
            raise StorageFailure("snapshot_resource_limit", identifier=snapshot_id)
        try:
            free_bytes = shutil.disk_usage(root).free
        except OSError as error:
            raise StorageFailure("snapshot_space_preflight_failed") from error
        if free_bytes < max(source_size * 2, 1):
            raise StorageFailure("snapshot_space_insufficient", identifier=snapshot_id)

        final_database = root / f"{snapshot_id}.sqlite3"
        final_manifest = root / f"{snapshot_id}.manifest.json"
        staging_database = root / f".{snapshot_id}.sqlite3.partial"
        staging_manifest = root / f".{snapshot_id}.manifest.json.partial"
        claimed_paths = (
            final_database,
            final_manifest,
            staging_database,
            staging_manifest,
        )
        if any(path.exists() or path.is_symlink() for path in claimed_paths):
            raise StorageFailure("snapshot_target_exists", identifier=snapshot_id)

        started = monotonic()
        completed_pages = False
        published: list[Path] = []
        try:
            _claim_private_file(staging_database)
            source_connection = sqlite3.connect(
                f"file:{source.as_posix()}?mode=ro",
                uri=True,
                isolation_level=None,
                timeout=min(timeout_seconds, 5.0),
            )
            target_connection = sqlite3.connect(
                staging_database,
                isolation_level=None,
                timeout=min(timeout_seconds, 5.0),
            )
            try:
                source_connection.execute(
                    f"PRAGMA busy_timeout = {max(1, int(min(timeout_seconds, 5.0) * 1000))}"
                )

                def progress(status: int, remaining: int, total: int) -> None:
                    nonlocal completed_pages
                    if monotonic() - started > timeout_seconds:
                        raise StorageFailure("snapshot_timeout", identifier=snapshot_id)
                    if total < 0 or remaining < 0 or remaining > total:
                        raise StorageFailure(
                            "snapshot_incomplete", identifier=snapshot_id
                        )
                    if total * 4096 > max_database_bytes:
                        raise StorageFailure(
                            "snapshot_resource_limit", identifier=snapshot_id
                        )
                    if fault_hook is not None:
                        fault_hook("backup_progress")
                    completed_pages = status == sqlite3.SQLITE_DONE and remaining == 0

                source_connection.backup(
                    target_connection,
                    pages=pages_per_step,
                    progress=progress,
                    sleep=busy_sleep_seconds,
                )
            finally:
                target_connection.close()
                source_connection.close()
            if not completed_pages:
                raise StorageFailure("snapshot_incomplete", identifier=snapshot_id)
            _fsync_file(staging_database)

            database_sha256 = _file_sha256(staging_database)
            database_size = staging_database.stat().st_size
            manifest = self._build_snapshot_manifest(
                staging_database,
                snapshot_id=snapshot_id,
                database_sha256=database_sha256,
                database_size=database_size,
            )
            manifest_bytes = _canonical_json_bytes(manifest)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            _write_private_file(staging_manifest, manifest_bytes)
            if fault_hook is not None:
                fault_hook("before_publish")

            os.replace(staging_database, final_database)
            published.append(final_database)
            os.replace(staging_manifest, final_manifest)
            published.append(final_manifest)
            _fsync_directory(root)
            return SnapshotResult(
                status=SnapshotStatus.PUBLISHED,
                snapshot_id=snapshot_id,
                database_path=final_database,
                manifest_path=final_manifest,
                database_sha256=database_sha256,
                manifest_sha256=manifest_sha256,
                database_size=database_size,
            )
        except StorageFailure:
            for path in (*published, staging_database, staging_manifest):
                _unlink_owned_file(path)
            raise
        except (OSError, sqlite3.Error) as error:
            for path in (*published, staging_database, staging_manifest):
                _unlink_owned_file(path)
            raise StorageFailure(
                "snapshot_io_failed", identifier=snapshot_id
            ) from error

    def restore_snapshot(
        self,
        snapshot_database: Path,
        snapshot_manifest: Path,
        *,
        expected_manifest_sha256: str,
        restore_root: Path,
    ) -> RestoreResult:
        """Restore only into a new root and require the exact full gate."""
        database, manifest_path, target = self._validate_restore_paths(
            snapshot_database, snapshot_manifest, restore_root
        )
        active_before = _file_sha256(self.repository.database_path)
        manifest_bytes = _read_regular_file(manifest_path, max_bytes=4 * 1024 * 1024)
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if not hmac.compare_digest(actual_manifest_sha256, expected_manifest_sha256):
            raise StorageFailure("restore_manifest_hash_mismatch")
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StorageFailure("restore_manifest_invalid") from error
        if not isinstance(manifest, dict):
            raise StorageFailure("restore_manifest_invalid")
        required_checks = manifest.get("required_checks")
        if (
            not isinstance(required_checks, list)
            or tuple(required_checks) != self.REQUIRED_RESTORE_CHECKS
        ):
            raise StorageFailure("restore_check_set_invalid")
        expected_database_sha256 = manifest.get("database_sha256")
        if not isinstance(expected_database_sha256, str) or not hmac.compare_digest(
            _file_sha256(database), expected_database_sha256
        ):
            raise StorageFailure("restore_database_hash_mismatch")

        target.mkdir(mode=0o700)
        restored_database = target / "aura.sqlite3"
        try:
            _copy_private_file(database, restored_database)
            if not hmac.compare_digest(
                _file_sha256(restored_database), expected_database_sha256
            ):
                raise StorageFailure("restore_database_hash_mismatch")
            connection = sqlite3.connect(restored_database, isolation_level=None)
            try:
                integrity = tuple(
                    str(row[0])
                    for row in connection.execute("PRAGMA integrity_check").fetchall()
                )
                if integrity != ("ok",):
                    raise StorageFailure("restore_integrity_failed")
                foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
                if foreign_keys:
                    raise StorageFailure("restore_foreign_key_failed")
                schema_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                if schema_version != int(manifest.get("sqlite_schema", -1)):
                    raise StorageFailure("restore_schema_mismatch")
                table_counts, table_digests = _canonical_table_facts(connection)
                manifest_counts = _string_int_map(manifest.get("table_counts"))
                manifest_digests = _string_map(manifest.get("table_digests"))
                if table_counts != manifest_counts:
                    raise StorageFailure("restore_table_count_mismatch")
                if table_digests != manifest_digests:
                    raise StorageFailure("restore_table_digest_mismatch")
                rebuild_fts(connection)
                connection.commit()
                fts_counts, fts_digests = _fts_facts(connection)
                if fts_counts != _string_int_map(manifest.get("fts_counts")):
                    raise StorageFailure("restore_fts_count_mismatch")
                if fts_digests != _string_map(manifest.get("fts_digests")):
                    raise StorageFailure("restore_fts_digest_mismatch")
            finally:
                connection.close()

            restored_repository = StorageRepository(restored_database)
            projection = self.projection_factory(
                target / "projection-generations", restored_repository
            )
            generation = projection.rebuild(
                generation_id=f"restore-{expected_database_sha256[:16]}",
                created_at="1970-01-01T00:00:00Z",
            )
            expected_origins = restored_repository.projection_origin_ids()
            expected_ids_sha256 = hashlib.sha256(
                json.dumps(expected_origins, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                generation.origin_count != len(expected_origins)
                or generation.origin_ids_sha256 != expected_ids_sha256
                or generation.metric != "cosine"
            ):
                raise StorageFailure("restore_projection_parity_failed")

            fixture_results = dict(
                self.fixture_verifier(restored_repository, projection)
            )
            if tuple(fixture_results) != self.FIXTURE_CHECKS or not all(
                value is True for value in fixture_results.values()
            ):
                raise StorageFailure("restore_fixture_failed")
            active_after = _file_sha256(self.repository.database_path)
            if not hmac.compare_digest(active_before, active_after):
                raise StorageFailure("restore_active_source_changed")
            return RestoreResult(
                status=RestoreStatus.COMPLETE,
                database_path=restored_database,
                checks=self.REQUIRED_RESTORE_CHECKS,
                table_counts=table_counts,
                manifest_table_counts=manifest_counts,
                table_digests=table_digests,
                manifest_table_digests=manifest_digests,
                projection_origin_count=generation.origin_count,
                projection_ids_sha256=generation.origin_ids_sha256,
            )
        except StorageFailure:
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            raise StorageFailure("restore_verification_failed") from error

    def export_scope_json(
        self,
        output_root: Path,
        *,
        scope_id: str,
        export_id: str,
        output_format: str = "json",
        created_at: str | None = None,
        fault_hook: SnapshotFaultHook | None = None,
    ) -> ExportResult:
        """Atomically publish actual allowlisted rows for one exact scope."""
        try:
            safe_export_format(output_format)
        except StoragePathError as error:
            raise StorageFailure("export_format_unsupported") from error
        try:
            safe_storage_component(scope_id)
        except StoragePathError as error:
            raise StorageFailure("export_scope_invalid") from error
        try:
            safe_storage_component(export_id)
        except StoragePathError as error:
            raise StorageFailure("export_id_invalid") from error
        root = _resolve_directory(output_root, "export_destination")
        final_path = root / f"{export_id}.json"
        staging_path = root / f".{export_id}.json.partial"
        if any(
            path.exists() or path.is_symlink() for path in (final_path, staging_path)
        ):
            raise StorageFailure("export_target_exists", identifier=export_id)

        records = self._collect_export_records(scope_id)
        counts = {name: len(records[name]) for name in _EXPORT_RECORDS}
        record_hashes = {
            name: hashlib.sha256(_canonical_json_bytes(records[name])).hexdigest()
            for name in _EXPORT_RECORDS
        }
        payload: dict[str, object] = {
            "schema_version": 1,
            "scope_id": scope_id,
            "export_id": export_id,
            "created_at": created_at
            or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "format": "json",
            "counts": counts,
            "record_hashes": record_hashes,
            "records": records,
        }
        manifest_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        payload["manifest_sha256"] = manifest_sha256
        try:
            _write_private_file(staging_path, _canonical_json_bytes(payload))
            if fault_hook is not None:
                fault_hook("before_publish")
            os.replace(staging_path, final_path)
            _fsync_directory(root)
        except StorageFailure:
            _unlink_owned_file(staging_path)
            _unlink_owned_file(final_path)
            raise
        except OSError as error:
            _unlink_owned_file(staging_path)
            _unlink_owned_file(final_path)
            raise StorageFailure("export_io_failed", identifier=export_id) from error
        return ExportResult(
            status=ExportStatus.PUBLISHED,
            path=final_path,
            manifest_sha256=manifest_sha256,
            counts=counts,
            record_hashes=record_hashes,
        )

    def verify_scope_export(
        self, export_path: Path, *, expected_scope_id: str
    ) -> ExportVerification:
        """Verify manifest, record, ID, count, and live-ledger round-trip parity."""
        path = _resolve_regular_file(export_path, "export_verification")
        payload_bytes = _read_regular_file(path, max_bytes=256 * 1024 * 1024)
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StorageFailure("export_manifest_invalid") from error
        if (
            not isinstance(payload, dict)
            or payload.get("scope_id") != expected_scope_id
        ):
            raise StorageFailure("export_scope_mismatch")
        supplied_manifest = payload.pop("manifest_sha256", None)
        if not isinstance(supplied_manifest, str) or not hmac.compare_digest(
            hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
            supplied_manifest,
        ):
            raise StorageFailure("export_manifest_hash_mismatch")
        records_value = payload.get("records")
        if (
            not isinstance(records_value, dict)
            or set(records_value) != set(_EXPORT_RECORDS)
            or any(
                not isinstance(records_value[name], list) for name in _EXPORT_RECORDS
            )
        ):
            raise StorageFailure("export_record_shape_invalid")
        exported_records = {name: list(records_value[name]) for name in _EXPORT_RECORDS}
        exported_counts = {
            name: len(exported_records[name]) for name in _EXPORT_RECORDS
        }
        exported_hashes = {
            name: hashlib.sha256(
                _canonical_json_bytes(exported_records[name])
            ).hexdigest()
            for name in _EXPORT_RECORDS
        }
        if exported_counts != _string_int_map(payload.get("counts")):
            raise StorageFailure("export_count_mismatch")
        if exported_hashes != _string_map(payload.get("record_hashes")):
            raise StorageFailure("export_record_hash_mismatch")
        current_records = self._collect_export_records(expected_scope_id)
        current_counts = {name: len(current_records[name]) for name in _EXPORT_RECORDS}
        current_hashes = {
            name: hashlib.sha256(
                _canonical_json_bytes(current_records[name])
            ).hexdigest()
            for name in _EXPORT_RECORDS
        }
        if exported_counts != current_counts or exported_hashes != current_hashes:
            raise StorageFailure("export_round_trip_mismatch")
        return ExportVerification(
            valid=True,
            counts=exported_counts,
            record_hashes=exported_hashes,
        )

    def plan_deletion(
        self,
        *,
        action: str,
        scope_id: str,
        target_id: str | None = None,
        ttl_seconds: float = 300.0,
        restore_proof: RestoreProof | None = None,
    ) -> DeletionPlan:
        """Inventory one exact destructive action without mutating any target."""
        try:
            deletion_action = DeletionAction(action)
        except ValueError as error:
            raise StorageFailure("deletion_action_invalid") from error
        try:
            safe_storage_component(scope_id)
        except StoragePathError as error:
            raise StorageFailure("deletion_scope_invalid") from error
        if target_id is not None:
            try:
                safe_storage_component(target_id)
            except StoragePathError as error:
                raise StorageFailure("deletion_target_id_invalid") from error
        if deletion_action is DeletionAction.SESSION and target_id is None:
            raise StorageFailure("deletion_session_required")
        if deletion_action in _TARGET_ACTION_KIND and target_id is None:
            raise StorageFailure("deletion_target_id_required")
        if deletion_action in {
            DeletionAction.ARCHIVE,
            DeletionAction.BACKUP_GENERATION,
        }:
            if not _valid_restore_proof(restore_proof, self.REQUIRED_RESTORE_CHECKS):
                raise StorageFailure("deletion_restore_proof_required")
        elif restore_proof is not None:
            raise StorageFailure("deletion_restore_proof_unexpected")
        if not 0 < ttl_seconds <= _DELETION_MAX_TTL_SECONDS:
            raise StorageFailure("deletion_ttl_invalid")

        canonical_items = self._canonical_deletion_inventory(
            deletion_action, scope_id, target_id
        )
        target_items = self._active_deletion_inventory(
            deletion_action, scope_id, target_id
        )
        items = tuple(
            sorted(
                (*canonical_items, *target_items),
                key=lambda item: (item.target, item.identity, item.content_sha256),
            )
        )
        if deletion_action in _TARGET_ACTION_KIND and not target_items:
            raise StorageFailure("deletion_target_unavailable", identifier=target_id)
        counts = _deletion_counts(items)
        inventory_digest = _deletion_inventory_digest(items)
        now = float(self._deletion_clock())
        base = {
            "plan_id": os.urandom(16).hex(),
            "action": deletion_action.value,
            "scope_id": scope_id,
            "target_id": target_id,
            "policy_version": _DELETION_POLICY_VERSION,
            "created_at": now,
            "expires_at": now + ttl_seconds,
            "inventory_digest": inventory_digest,
            "counts": counts,
            "items": [_deletion_item_payload(item) for item in items],
            "retained_copies": [
                {"kind": copy.kind, "alias": copy.alias}
                for copy in self._retained_copies
            ],
            "restore_proof_generation": (
                restore_proof.generation_id if restore_proof is not None else None
            ),
        }
        challenge = _deletion_hmac(self._deletion_secret, base)
        return DeletionPlan(
            plan_id=str(base["plan_id"]),
            action=deletion_action,
            scope_id=scope_id,
            target_id=target_id,
            policy_version=_DELETION_POLICY_VERSION,
            created_at=now,
            expires_at=now + ttl_seconds,
            inventory_digest=inventory_digest,
            counts=counts,
            items=items,
            retained_copies=self._retained_copies,
            restore_proof_generation=base["restore_proof_generation"],
            challenge=challenge,
        )

    def confirm_deletion(
        self,
        plan: DeletionPlan,
        *,
        challenge: str,
        scope_id: str,
    ) -> DeletionConfirmation:
        """Confirm one unexpired untampered plan without executing it."""
        plan_digest = self._validate_deletion_plan(plan, check_expiry=True)
        if scope_id != plan.scope_id:
            raise StorageFailure("deletion_scope_mismatch")
        if not hmac.compare_digest(challenge, plan.challenge):
            raise StorageFailure("deletion_confirmation_invalid")
        confirmation_id = _deletion_hmac(
            self._deletion_secret,
            {
                "kind": "confirmed-deletion",
                "plan_digest": plan_digest,
                "expires_at": plan.expires_at,
            },
        )
        if confirmation_id in self._consumed_confirmations:
            raise StorageFailure("deletion_confirmation_replayed")
        self._confirmed_deletions[confirmation_id] = plan_digest
        return DeletionConfirmation(
            confirmation_id=confirmation_id,
            plan_digest=plan_digest,
            expires_at=plan.expires_at,
        )

    def execute_deletion(
        self,
        plan: DeletionPlan,
        confirmation: DeletionConfirmation,
    ) -> DeletionExecution:
        """Consume confirmation, commit exact SQLite rows, then active targets."""
        plan_digest = self._validate_deletion_plan(plan, check_expiry=True)
        if confirmation.confirmation_id in self._consumed_confirmations:
            raise StorageFailure("deletion_confirmation_replayed")
        expected_confirmation = self._confirmed_deletions.get(
            confirmation.confirmation_id
        )
        if (
            expected_confirmation is None
            or expected_confirmation != plan_digest
            or confirmation.plan_digest != plan_digest
            or confirmation.expires_at != plan.expires_at
        ):
            raise StorageFailure("deletion_confirmation_invalid")
        self._consumed_confirmations.add(confirmation.confirmation_id)
        self._confirmed_deletions.pop(confirmation.confirmation_id, None)

        planned_canonical = tuple(
            item for item in plan.items if item.target.startswith("sqlite:")
        )
        planned_targets = tuple(
            item for item in plan.items if not item.target.startswith("sqlite:")
        )
        current_targets = self._active_deletion_inventory(
            plan.action, plan.scope_id, plan.target_id
        )
        if current_targets != planned_targets:
            raise StorageFailure("deletion_inventory_changed")

        connection = open_database(self.repository.database_path)
        canonical_committed = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_canonical = self._canonical_deletion_inventory(
                plan.action,
                plan.scope_id,
                plan.target_id,
                connection=connection,
            )
            if current_canonical != planned_canonical:
                raise StorageFailure("deletion_inventory_changed")
            self._delete_canonical_items(connection, planned_canonical)
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise StorageFailure("deletion_foreign_key_failed")
            connection.commit()
            canonical_committed = True
        except StorageFailure:
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise StorageFailure("deletion_sqlite_failed") from error
        finally:
            connection.close()

        attempted: list[str] = []
        failed: list[str] = []
        for target in self._selected_deletion_targets(
            plan.action, plan.scope_id, plan.target_id
        ):
            identities = tuple(
                item.identity for item in planned_targets if item.target == target.name
            )
            if not identities:
                continue
            attempted.append(target.name)
            try:
                target.delete(identities)
                if target.kind == "projection" and target.reconcile is not None:
                    target.reconcile()
            except Exception:
                failed.append(target.name)
        attempted_tuple = tuple(sorted(attempted))
        failed_tuple = tuple(sorted(failed))
        execution_id = _deletion_hmac(
            self._deletion_secret,
            {
                "kind": "deletion-execution",
                "plan_digest": plan_digest,
                "canonical_committed": canonical_committed,
                "attempted_targets": attempted_tuple,
                "failed_targets": failed_tuple,
            },
        )
        return DeletionExecution(
            execution_id=execution_id,
            plan_digest=plan_digest,
            canonical_committed=canonical_committed,
            attempted_targets=attempted_tuple,
            failed_targets=failed_tuple,
        )

    def verify_deletion(
        self,
        plan: DeletionPlan,
        execution: DeletionExecution,
    ) -> DeletionResult:
        """Re-query every approved exact identity and report all residual truth."""
        plan_digest = self._validate_deletion_plan(plan, check_expiry=False)
        expected_execution_id = _deletion_hmac(
            self._deletion_secret,
            {
                "kind": "deletion-execution",
                "plan_digest": plan_digest,
                "canonical_committed": execution.canonical_committed,
                "attempted_targets": execution.attempted_targets,
                "failed_targets": execution.failed_targets,
            },
        )
        if execution.plan_digest != plan_digest or not hmac.compare_digest(
            execution.execution_id, expected_execution_id
        ):
            raise StorageFailure("deletion_execution_tampered")

        planned_canonical = tuple(
            item for item in plan.items if item.target.startswith("sqlite:")
        )
        remaining_canonical = self._remaining_canonical_items(planned_canonical)
        retry_targets: set[str] = set()
        remaining_counts = _deletion_counts(remaining_canonical)
        if remaining_canonical or not execution.canonical_committed:
            retry_targets.add("canonical_sqlite")

        targets_by_name = {
            target.name: target
            for target in self._selected_deletion_targets(
                plan.action, plan.scope_id, plan.target_id
            )
        }
        failed_targets = set(execution.failed_targets)
        for target_name in sorted(
            {
                item.target
                for item in plan.items
                if not item.target.startswith("sqlite:")
            }
        ):
            identities = tuple(
                item.identity for item in plan.items if item.target == target_name
            )
            target = targets_by_name.get(target_name)
            if target is None:
                failed_targets.add(target_name)
                retry_targets.add(target_name)
                remaining_counts[target_name] = len(identities)
                continue
            try:
                remaining = dict(target.remaining(identities))
            except Exception:
                failed_targets.add(target_name)
                retry_targets.add(target_name)
                remaining_counts[target_name] = len(identities)
                continue
            unexpected = set(remaining) - set(identities)
            if unexpected or any(
                not _valid_content_digest(identity, digest)
                for identity, digest in remaining.items()
            ):
                failed_targets.add(target_name)
                retry_targets.add(target_name)
                remaining_counts[target_name] = len(identities)
                continue
            if remaining:
                retry_targets.add(target_name)
                remaining_counts[target_name] = len(remaining)

        retry_targets.update(failed_targets)
        deleted_counts = {
            target: max(0, count - remaining_counts.get(target, 0))
            for target, count in plan.counts.items()
        }
        complete = (
            execution.canonical_committed and not failed_targets and not retry_targets
        )
        return DeletionResult(
            status=(DeletionStatus.COMPLETE if complete else DeletionStatus.INCOMPLETE),
            action=plan.action,
            scope_id=plan.scope_id,
            inventory_digest=plan.inventory_digest,
            deleted_counts=deleted_counts,
            failed_targets=tuple(sorted(failed_targets)),
            retry_targets=tuple(sorted(retry_targets)),
            retained_copies=self._retained_after_action(plan),
            application_level_deletion=execution.canonical_committed,
            physical_purge="best_effort",
            forensic_erasure=False,
        )

    def _validate_deletion_plan(self, plan: DeletionPlan, *, check_expiry: bool) -> str:
        if plan.policy_version != _DELETION_POLICY_VERSION:
            raise StorageFailure("deletion_plan_tampered")
        if plan.counts != _deletion_counts(plan.items) or not hmac.compare_digest(
            plan.inventory_digest, _deletion_inventory_digest(plan.items)
        ):
            raise StorageFailure("deletion_plan_tampered")
        plan_digest = _deletion_hmac(
            self._deletion_secret, _deletion_plan_payload(plan)
        )
        if not hmac.compare_digest(plan.challenge, plan_digest):
            raise StorageFailure("deletion_plan_tampered")
        if check_expiry and float(self._deletion_clock()) > plan.expires_at:
            raise StorageFailure("deletion_plan_expired")
        return plan_digest

    def _selected_deletion_targets(
        self,
        action: DeletionAction,
        scope_id: str,
        target_id: str | None,
    ) -> tuple[DeletionTarget, ...]:
        del scope_id, target_id
        if action in {DeletionAction.SCOPE, DeletionAction.SESSION}:
            return tuple(
                target
                for target in self._deletion_targets
                if target.kind in _ACTIVE_DELETION_KINDS
            )
        required_kind = _TARGET_ACTION_KIND.get(action)
        return tuple(
            target for target in self._deletion_targets if target.kind == required_kind
        )

    def _active_deletion_inventory(
        self,
        action: DeletionAction,
        scope_id: str,
        target_id: str | None,
    ) -> tuple[DeletionItem, ...]:
        items: list[DeletionItem] = []
        for target in self._selected_deletion_targets(action, scope_id, target_id):
            try:
                inventory = dict(target.inventory(action, scope_id, target_id))
            except Exception as error:
                raise StorageFailure(
                    "deletion_inventory_failed", identifier=target.name
                ) from error
            for identity, digest in sorted(inventory.items()):
                if not _valid_content_digest(identity, digest):
                    raise StorageFailure(
                        "deletion_inventory_invalid", identifier=target.name
                    )
                items.append(DeletionItem(target.name, identity, digest))
        return tuple(sorted(items, key=lambda item: (item.target, item.identity)))

    def _canonical_deletion_inventory(
        self,
        action: DeletionAction,
        scope_id: str,
        target_id: str | None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[DeletionItem, ...]:
        """Return exact SQLite and external-content FTS rows for one action."""
        if action not in {DeletionAction.SCOPE, DeletionAction.SESSION}:
            return ()
        owns_connection = connection is None
        if connection is None:
            connection = sqlite3.connect(
                f"file:{self.repository.database_path.as_posix()}?mode=ro",
                uri=True,
                isolation_level=None,
            )
        previous_factory = connection.row_factory
        connection.row_factory = sqlite3.Row
        try:
            items: dict[tuple[str, str], DeletionItem] = {}

            def add_rows(
                table: str,
                statement: str,
                parameters: tuple[object, ...],
                *,
                primary_keys: tuple[str, ...] | None = None,
            ) -> tuple[sqlite3.Row, ...]:
                rows = tuple(connection.execute(statement, parameters).fetchall())
                keys = primary_keys or _DELETION_PRIMARY_KEYS[table]
                for row in rows:
                    item = _deletion_item_from_row(table, keys, row)
                    items[(item.target, item.identity)] = item
                return rows

            if action is DeletionAction.SCOPE:
                scoped_tables = (
                    "memory_scopes",
                    "sessions",
                    "turns",
                    "events",
                    "derived_memories",
                    "memory_sources",
                    "memory_supersessions",
                    "memory_retractions",
                    "profile_versions",
                    "retrieval_runs",
                    "legacy_fragments",
                )
                for table in scoped_tables:
                    add_rows(
                        table,
                        f'SELECT * FROM "{table}" WHERE scope_id=?',
                        (scope_id,),
                    )
                add_rows(
                    "retrieval_candidates",
                    "SELECT candidate.* FROM retrieval_candidates AS candidate "
                    "JOIN retrieval_runs AS run ON run.run_id=candidate.run_id "
                    "WHERE run.scope_id=?",
                    (scope_id,),
                )
                add_rows(
                    "event_fts",
                    "SELECT fts.rowid,fts.content FROM event_fts AS fts "
                    "JOIN events AS event ON event.event_pk=fts.rowid "
                    "WHERE event.scope_id=?",
                    (scope_id,),
                    primary_keys=("rowid",),
                )
                add_rows(
                    "memory_fts",
                    "SELECT fts.rowid,fts.canonical_text FROM memory_fts AS fts "
                    "JOIN derived_memories AS memory ON memory.memory_pk=fts.rowid "
                    "WHERE memory.scope_id=?",
                    (scope_id,),
                    primary_keys=("rowid",),
                )
                origin_rows = connection.execute(
                    "SELECT event_id AS origin_id FROM events WHERE scope_id=? "
                    "UNION SELECT memory_id FROM derived_memories WHERE scope_id=?",
                    (scope_id, scope_id),
                ).fetchall()
            else:
                assert target_id is not None
                session_rows = add_rows(
                    "sessions",
                    "SELECT * FROM sessions WHERE scope_id=? AND session_id=?",
                    (scope_id, target_id),
                )
                if not session_rows:
                    raise StorageFailure(
                        "deletion_target_not_found", identifier=target_id
                    )
                add_rows(
                    "turns",
                    "SELECT * FROM turns WHERE scope_id=? AND session_id=?",
                    (scope_id, target_id),
                )
                add_rows(
                    "events",
                    "SELECT event.* FROM events AS event JOIN turns AS turn "
                    "ON turn.turn_id=event.turn_id AND turn.scope_id=event.scope_id "
                    "WHERE turn.scope_id=? AND turn.session_id=?",
                    (scope_id, target_id),
                )
                memory_predicate = (
                    "memory.scope_id=? AND (EXISTS (SELECT 1 FROM events AS primary_event "
                    "JOIN turns AS primary_turn ON primary_turn.turn_id=primary_event.turn_id "
                    "AND primary_turn.scope_id=primary_event.scope_id "
                    "WHERE primary_event.event_id=memory.primary_source_event_id "
                    "AND primary_event.scope_id=memory.scope_id "
                    "AND primary_turn.session_id=?) OR EXISTS (SELECT 1 FROM memory_sources AS source "
                    "JOIN events AS source_event ON source_event.event_id=source.event_id "
                    "AND source_event.scope_id=source.scope_id JOIN turns AS source_turn "
                    "ON source_turn.turn_id=source_event.turn_id "
                    "AND source_turn.scope_id=source_event.scope_id "
                    "WHERE source.memory_id=memory.memory_id AND source.scope_id=memory.scope_id "
                    "AND source_turn.session_id=?))"
                )
                memory_rows = add_rows(
                    "derived_memories",
                    f"SELECT memory.* FROM derived_memories AS memory WHERE {memory_predicate}",
                    (scope_id, target_id, target_id),
                )
                event_rows = connection.execute(
                    "SELECT event.event_id,event.event_pk FROM events AS event "
                    "JOIN turns AS turn ON turn.turn_id=event.turn_id "
                    "AND turn.scope_id=event.scope_id "
                    "WHERE turn.scope_id=? AND turn.session_id=?",
                    (scope_id, target_id),
                ).fetchall()
                event_ids = {str(row["event_id"]) for row in event_rows}
                memory_ids = {str(row["memory_id"]) for row in memory_rows}
                for memory_id in sorted(memory_ids):
                    add_rows(
                        "memory_sources",
                        "SELECT * FROM memory_sources WHERE scope_id=? AND memory_id=?",
                        (scope_id, memory_id),
                    )
                    add_rows(
                        "memory_supersessions",
                        "SELECT * FROM memory_supersessions WHERE scope_id=? "
                        "AND (old_memory_id=? OR new_memory_id=?)",
                        (scope_id, memory_id, memory_id),
                    )
                    add_rows(
                        "memory_retractions",
                        "SELECT * FROM memory_retractions WHERE scope_id=? AND memory_id=?",
                        (scope_id, memory_id),
                    )
                for event_id in sorted(event_ids):
                    add_rows(
                        "memory_supersessions",
                        "SELECT * FROM memory_supersessions WHERE scope_id=? "
                        "AND basis_event_id=?",
                        (scope_id, event_id),
                    )
                    add_rows(
                        "memory_retractions",
                        "SELECT * FROM memory_retractions WHERE scope_id=? "
                        "AND basis_event_id=?",
                        (scope_id, event_id),
                    )
                for row in event_rows:
                    add_rows(
                        "event_fts",
                        "SELECT rowid,content FROM event_fts WHERE rowid=?",
                        (int(row["event_pk"]),),
                        primary_keys=("rowid",),
                    )
                for row in memory_rows:
                    add_rows(
                        "memory_fts",
                        "SELECT rowid,canonical_text FROM memory_fts WHERE rowid=?",
                        (int(row["memory_pk"]),),
                        primary_keys=("rowid",),
                    )
                origin_rows = tuple(
                    {"origin_id": identity}
                    for identity in sorted(event_ids | memory_ids)
                )

            for row in origin_rows:
                origin_id = str(row["origin_id"])
                for candidate in (
                    origin_id,
                    f"event:{origin_id}",
                    f"memory:{origin_id}",
                ):
                    add_rows(
                        "legacy_sources",
                        "SELECT * FROM legacy_sources WHERE imported_origin_id=?",
                        (candidate,),
                    )
            return tuple(
                sorted(items.values(), key=lambda item: (item.target, item.identity))
            )
        except sqlite3.Error as error:
            raise StorageFailure("deletion_inventory_failed") from error
        finally:
            connection.row_factory = previous_factory
            if owns_connection:
                connection.close()

    def _delete_canonical_items(
        self,
        connection: sqlite3.Connection,
        items: tuple[DeletionItem, ...],
    ) -> None:
        by_table: dict[str, list[DeletionItem]] = {}
        for item in items:
            table = item.target.removeprefix("sqlite:")
            if table in _FTS_TABLES:
                continue
            if table not in _DELETION_PRIMARY_KEYS:
                raise StorageFailure("deletion_plan_tampered")
            by_table.setdefault(table, []).append(item)
        for table in _CANONICAL_DELETE_ORDER:
            keys = _DELETION_PRIMARY_KEYS[table]
            where = " AND ".join(f'"{key}"=?' for key in keys)
            for item in by_table.get(table, []):
                values = _deletion_identity_values(item.identity, len(keys))
                cursor = connection.execute(
                    f'DELETE FROM "{table}" WHERE {where}', values
                )
                if cursor.rowcount != 1:
                    raise StorageFailure("deletion_inventory_changed")

    def _remaining_canonical_items(
        self, items: tuple[DeletionItem, ...]
    ) -> tuple[DeletionItem, ...]:
        connection = sqlite3.connect(
            f"file:{self.repository.database_path.as_posix()}?mode=ro",
            uri=True,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        remaining: list[DeletionItem] = []
        try:
            for item in items:
                table = item.target.removeprefix("sqlite:")
                keys = (
                    ("rowid",)
                    if table in _FTS_TABLES
                    else _DELETION_PRIMARY_KEYS.get(table)
                )
                if keys is None:
                    raise StorageFailure("deletion_plan_tampered")
                values = _deletion_identity_values(item.identity, len(keys))
                where = " AND ".join(f'"{key}"=?' for key in keys)
                row = connection.execute(
                    f'SELECT * FROM "{table}" WHERE {where}', values
                ).fetchone()
                if row is not None:
                    remaining.append(_deletion_item_from_row(table, keys, row))
            return tuple(
                sorted(remaining, key=lambda item: (item.target, item.identity))
            )
        except sqlite3.Error as error:
            raise StorageFailure("deletion_verification_failed") from error
        finally:
            connection.close()

    def _retained_after_action(self, plan: DeletionPlan) -> tuple[ResidualCopy, ...]:
        """Report every configured copy; this plan never infers physical purge."""
        return plan.retained_copies

    def _collect_export_records(
        self, scope_id: str
    ) -> dict[str, list[dict[str, object]]]:
        connection = sqlite3.connect(
            f"file:{self.repository.database_path.as_posix()}?mode=ro",
            uri=True,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN")
            records: dict[str, list[dict[str, object]]] = {
                "sessions": _query_dicts(
                    connection,
                    "SELECT session_id,created_at,closed_at FROM sessions "
                    "WHERE scope_id=? ORDER BY created_at,session_id",
                    (scope_id,),
                ),
                "turns": _query_dicts(
                    connection,
                    "SELECT turn_id,session_id,idempotency_key,request_hash_version,"
                    "request_hash,response_hash,occurred_at FROM turns "
                    "WHERE scope_id=? ORDER BY occurred_at,turn_id",
                    (scope_id,),
                ),
                "events": _event_export_rows(connection, scope_id),
                "derived_memories": _query_dicts(
                    connection,
                    "SELECT memory_id,memory_kind,canonical_text,confidence,"
                    "epistemic_status,primary_source_event_id,created_at,content_sha256 "
                    "FROM derived_memories WHERE scope_id=? "
                    "ORDER BY created_at,memory_id",
                    (scope_id,),
                ),
                "memory_sources": _query_dicts(
                    connection,
                    "SELECT memory_id,event_id,relation FROM memory_sources "
                    "WHERE scope_id=? ORDER BY memory_id,event_id",
                    (scope_id,),
                ),
                "memory_supersessions": _query_dicts(
                    connection,
                    "SELECT old_memory_id,new_memory_id,basis_event_id,reason,created_at "
                    "FROM memory_supersessions WHERE scope_id=? "
                    "ORDER BY created_at,old_memory_id,new_memory_id",
                    (scope_id,),
                ),
                "memory_retractions": _query_dicts(
                    connection,
                    "SELECT memory_id,basis_event_id,reason,created_at "
                    "FROM memory_retractions WHERE scope_id=? "
                    "ORDER BY created_at,memory_id",
                    (scope_id,),
                ),
                "profile_versions": _profile_export_rows(connection, scope_id),
            }
            connection.rollback()
            return records
        except sqlite3.Error as error:
            connection.rollback()
            raise StorageFailure("export_read_failed") from error
        finally:
            connection.close()

    def _build_snapshot_manifest(
        self,
        database_path: Path,
        *,
        snapshot_id: str,
        database_sha256: str,
        database_size: int,
    ) -> dict[str, object]:
        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            if tuple(str(row[0]) for row in integrity) != ("ok",):
                raise StorageFailure(
                    "snapshot_integrity_failed", identifier=snapshot_id
                )
            counts, digests = _canonical_table_facts(connection)
            fts_counts, fts_digests = _fts_facts(connection)
            schema_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            event_watermark = int(
                connection.execute(
                    "SELECT COALESCE(MAX(event_pk), 0) FROM events"
                ).fetchone()[0]
            )
            memory_watermark = int(
                connection.execute(
                    "SELECT COALESCE(MAX(memory_pk), 0) FROM derived_memories"
                ).fetchone()[0]
            )
            projection_row = connection.execute(
                "SELECT generation_id,embedding_model,config_sha256,metric,"
                "sqlite_watermark FROM projection_generations "
                "WHERE build_status='ready' "
                "ORDER BY completed_at DESC,generation_id DESC LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        projection_generation: dict[str, object] | None = None
        if projection_row is not None:
            projection_generation = {
                "generation_id": str(projection_row[0]),
                "embedding_model": str(projection_row[1]),
                "config_sha256": str(projection_row[2]),
                "metric": str(projection_row[3]),
                "sqlite_watermark": int(projection_row[4]),
            }
        return {
            "snapshot_format_version": _SNAPSHOT_FORMAT_VERSION,
            "config_version": _SNAPSHOT_CONFIG_VERSION,
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "tool_commit": self.tool_commit,
            "sqlite_schema": schema_version,
            "source_watermark": (event_watermark << 32) | memory_watermark,
            "table_counts": counts,
            "table_digests": digests,
            "fts_config": _FTS_CONFIG,
            "fts_counts": fts_counts,
            "fts_digests": fts_digests,
            "projection_generation": projection_generation,
            "database_size": database_size,
            "database_sha256": database_sha256,
            "required_checks": list(self.REQUIRED_RESTORE_CHECKS),
        }

    def _validate_snapshot_paths(
        self, destination_root: Path, snapshot_id: str
    ) -> tuple[Path, Path]:
        if not _SAFE_OPERATION_ID.fullmatch(snapshot_id):
            raise StorageFailure("snapshot_id_invalid")
        source = _resolve_regular_file(self.repository.database_path, "snapshot_source")
        root = _resolve_directory(destination_root, "snapshot_destination")
        if _paths_overlap(source, root):
            raise StorageFailure("snapshot_path_overlap")
        return source, root

    def _validate_restore_paths(
        self,
        snapshot_database: Path,
        snapshot_manifest: Path,
        restore_root: Path,
    ) -> tuple[Path, Path, Path]:
        database = _resolve_regular_file(snapshot_database, "restore_database")
        manifest = _resolve_regular_file(snapshot_manifest, "restore_manifest")
        if not restore_root.is_absolute():
            raise StorageFailure("restore_absolute_path_required")
        _reject_symlink_components(restore_root)
        target = restore_root.resolve(strict=False)
        if target.exists() or target.is_symlink():
            raise StorageFailure("restore_target_exists")
        if not target.parent.is_dir():
            raise StorageFailure("restore_parent_missing")
        active = self.repository.database_path.resolve(strict=True)
        if any(_paths_overlap(target, path) for path in (database, manifest, active)):
            raise StorageFailure("restore_path_overlap")
        return database, manifest, target


def _canonical_table_facts(
    connection: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, str]]:
    available = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table'"
        ).fetchall()
    }
    if not set(_CANONICAL_TABLES).issubset(available):
        raise StorageFailure("lifecycle_table_set_incomplete")
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for table in _CANONICAL_TABLES:
        columns = tuple(
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        )
        if not columns:
            raise StorageFailure("lifecycle_table_shape_invalid", identifier=table)
        quoted = ",".join(f'"{column}"' for column in columns)
        rows = connection.execute(
            f'SELECT {quoted} FROM "{table}" ORDER BY {quoted}'
        ).fetchall()
        counts[table] = len(rows)
        digests[table] = hashlib.sha256(
            b"\n".join(_canonical_json_bytes(list(row)) for row in rows)
        ).hexdigest()
    return counts, digests


def _fts_facts(
    connection: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, str]]:
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    columns_by_table = {
        "event_fts": "content",
        "memory_fts": "canonical_text",
    }
    for table in _FTS_TABLES:
        content_column = columns_by_table[table]
        rows = connection.execute(
            f'SELECT rowid,"{content_column}" FROM "{table}" ORDER BY rowid'
        ).fetchall()
        counts[table] = len(rows)
        digests[table] = hashlib.sha256(
            b"\n".join(_canonical_json_bytes(list(row)) for row in rows)
        ).hexdigest()
    return counts, digests


def _query_dicts(
    connection: sqlite3.Connection,
    statement: str,
    parameters: tuple[object, ...],
) -> list[dict[str, object]]:
    """Return deterministic allowlisted rows from a parameterized scope query."""
    return [dict(row) for row in connection.execute(statement, parameters).fetchall()]


def _event_export_rows(
    connection: sqlite3.Connection, scope_id: str
) -> list[dict[str, object]]:
    rows = connection.execute(
        "SELECT event_id,turn_id,ordinal,event_type,actor,observed_at,content,"
        "payload_json,content_sha256,source_kind FROM events WHERE scope_id=? "
        "ORDER BY observed_at,event_id",
        (scope_id,),
    ).fetchall()
    exported: list[dict[str, object]] = []
    for row in rows:
        values = dict(row)
        raw_payload = values.pop("payload_json")
        try:
            parsed = json.loads(str(raw_payload))
        except json.JSONDecodeError as error:
            raise StorageFailure("export_event_payload_invalid") from error
        sanitized = _sanitize_profile_value(parsed)
        values["payload"] = sanitized
        values["payload_sha256"] = hashlib.sha256(
            _canonical_json_bytes(sanitized)
        ).hexdigest()
        exported.append(values)
    return exported


def _profile_export_rows(
    connection: sqlite3.Connection, scope_id: str
) -> list[dict[str, object]]:
    rows = connection.execute(
        "SELECT profile_version,payload_json,created_at FROM profile_versions "
        "WHERE scope_id=? ORDER BY profile_version",
        (scope_id,),
    ).fetchall()
    exported: list[dict[str, object]] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as error:
            raise StorageFailure("export_profile_payload_invalid") from error
        sanitized = _sanitize_profile_value(payload)
        exported.append(
            {
                "profile_version": int(row["profile_version"]),
                "payload": sanitized,
                "payload_sha256": hashlib.sha256(
                    _canonical_json_bytes(sanitized)
                ).hexdigest(),
                "created_at": str(row["created_at"]),
            }
        )
    return exported


def _sanitize_profile_value(value: object) -> object:
    """Remove operational secrets and rebuildable vectors from portable JSON."""
    if isinstance(value, dict):
        return {
            str(key): _sanitize_profile_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).casefold().replace("-", "_") not in _PRIVATE_PROFILE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_profile_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise StorageFailure("export_payload_type_invalid")


def _valid_restore_proof(
    proof: RestoreProof | None, required_checks: tuple[str, ...]
) -> bool:
    return bool(
        proof is not None
        and _SAFE_OPERATION_ID.fullmatch(proof.generation_id)
        and re.fullmatch(r"[0-9a-f]{64}", proof.manifest_sha256)
        and proof.checks == required_checks
    )


def _valid_content_digest(identity: object, digest: object) -> bool:
    return bool(
        isinstance(identity, str)
        and identity
        and len(identity) <= 2048
        and "\x00" not in identity
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
    )


def _deletion_item_from_row(
    table: str,
    primary_keys: tuple[str, ...],
    row: sqlite3.Row,
) -> DeletionItem:
    identity = _canonical_json_bytes([row[key] for key in primary_keys]).decode("utf-8")
    content_sha256 = hashlib.sha256(_canonical_json_bytes(dict(row))).hexdigest()
    return DeletionItem(
        target=f"sqlite:{table}",
        identity=identity,
        content_sha256=content_sha256,
    )


def _deletion_identity_values(identity: str, expected: int) -> tuple[object, ...]:
    try:
        values = json.loads(identity)
    except json.JSONDecodeError as error:
        raise StorageFailure("deletion_plan_tampered") from error
    if not isinstance(values, list) or len(values) != expected:
        raise StorageFailure("deletion_plan_tampered")
    if any(value is None or isinstance(value, (dict, list, bool)) for value in values):
        raise StorageFailure("deletion_plan_tampered")
    return tuple(values)


def _deletion_item_payload(item: DeletionItem) -> dict[str, str]:
    return {
        "target": item.target,
        "identity": item.identity,
        "content_sha256": item.content_sha256,
    }


def _deletion_counts(items: tuple[DeletionItem, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.target] = counts.get(item.target, 0) + 1
    return {target: counts[target] for target in sorted(counts)}


def _deletion_inventory_digest(items: tuple[DeletionItem, ...]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes([_deletion_item_payload(item) for item in items])
    ).hexdigest()


def _deletion_plan_payload(plan: DeletionPlan) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "action": plan.action.value,
        "scope_id": plan.scope_id,
        "target_id": plan.target_id,
        "policy_version": plan.policy_version,
        "created_at": plan.created_at,
        "expires_at": plan.expires_at,
        "inventory_digest": plan.inventory_digest,
        "counts": plan.counts,
        "items": [_deletion_item_payload(item) for item in plan.items],
        "retained_copies": [
            {"kind": copy.kind, "alias": copy.alias} for copy in plan.retained_copies
        ],
        "restore_proof_generation": plan.restore_proof_generation,
    }


def _deletion_hmac(secret: bytes, payload: object) -> str:
    return hmac.new(secret, _canonical_json_bytes(payload), hashlib.sha256).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise StorageFailure("restore_manifest_invalid")
    return dict(value)


def _string_int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not isinstance(item, int)
        or isinstance(item, bool)
        or item < 0
        for key, item in value.items()
    ):
        raise StorageFailure("restore_manifest_invalid")
    return dict(value)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _resolve_regular_file(path: Path, code: str) -> Path:
    if not path.is_absolute():
        raise StorageFailure(f"{code}_absolute_path_required")
    try:
        _reject_symlink_components(path)
        resolved = path.resolve(strict=True)
        information = resolved.lstat()
    except FileNotFoundError as error:
        raise StorageFailure(f"{code}_missing") from error
    except OSError as error:
        raise StorageFailure(f"{code}_invalid") from error
    if not stat.S_ISREG(information.st_mode):
        raise StorageFailure(f"{code}_invalid")
    return resolved


def _resolve_directory(path: Path, code: str) -> Path:
    if not path.is_absolute():
        raise StorageFailure(f"{code}_absolute_path_required")
    try:
        _reject_symlink_components(path)
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise StorageFailure(f"{code}_missing") from error
    except OSError as error:
        raise StorageFailure(f"{code}_invalid") from error
    if not resolved.is_dir():
        raise StorageFailure(f"{code}_invalid")
    return resolved


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            information = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise StorageFailure("snapshot_path_invalid") from error
        if stat.S_ISLNK(information.st_mode):
            raise StorageFailure("snapshot_symlink_path")


def _claim_private_file(path: Path) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600, follow_symlinks=False)


def _write_private_file(path: Path, payload: bytes) -> None:
    _claim_private_file(path)
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_private_file(source: Path, destination: Path) -> None:
    _claim_private_file(destination)
    read_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    write_flags = (
        os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source, read_flags)
    destination_descriptor = os.open(destination, write_flags)
    with (
        os.fdopen(source_descriptor, "rb") as source_stream,
        os.fdopen(destination_descriptor, "wb") as destination_stream,
    ):
        shutil.copyfileobj(source_stream, destination_stream, length=_CHUNK_BYTES)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())
    _fsync_directory(destination.parent)


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    information = path.stat()
    if information.st_size > max_bytes:
        raise StorageFailure("restore_manifest_too_large")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read(max_bytes + 1)


def _file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_owned_file(path: Path) -> None:
    try:
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except OSError:
        pass
