"""Authorized, source-immutable import from disposable legacy Chroma copies."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import chromadb

from aura_backend.storage.models import StorageFailure
from aura_backend.storage.repository import StorageRepository

ClientFactory = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class LegacyImportAuthorization:
    """Exact evidence and path scope required before a legacy store can open."""

    source_alias: str
    root_fingerprint: str
    disposable_root: Path
    restore_workspace: Path
    allowlisted_sources: tuple[tuple[str, str], ...]
    evidence_status: str
    evidence_sha256: str
    forbidden_roots: tuple[Path, ...]
    anomaly_codes: tuple[str, ...] = ()

    def replace(self, **changes: Any) -> LegacyImportAuthorization:
        """Return a changed immutable authorization for adversarial checks."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class LegacyImportResult:
    """Content-free outcome for one exact source alias and fingerprint."""

    source_alias: str
    root_fingerprint: str
    collection_count: int
    records_seen: int
    imported_count: int
    existing_count: int
    fragment_count: int
    source_before_sha256: str
    source_after_sha256: str
    mapping_codes: tuple[str, ...]
    anomaly_counts: dict[str, int]


def legacy_tree_sha256(root: Path) -> str:
    """Hash relative paths and bytes of one synthetic/disposable source tree."""
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise StorageFailure("legacy_source_not_directory")
    digest = hashlib.sha256()
    entries = sorted(resolved.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise StorageFailure("legacy_source_symlink_rejected")
    for path in (item for item in entries if item.is_file()):
        relative = path.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


class LegacyImporter:
    """Read public Chroma pages only after exact disposable-copy authorization."""

    _known_metadata = frozenset({"user_id", "sender", "timestamp", "session_id"})

    def __init__(
        self,
        *,
        repository: StorageRepository,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.repository = repository
        self._client_factory = client_factory or chromadb.PersistentClient

    def import_authorized(
        self,
        authorization: LegacyImportAuthorization,
        *,
        page_size: int = 100,
    ) -> LegacyImportResult:
        """Import one exact restored root without mutating or repairing it."""
        if page_size <= 0 or page_size > 1_000:
            raise StorageFailure("invalid_legacy_page_size")
        source = self._validate_authorization(authorization)
        before = legacy_tree_sha256(source)
        if before != authorization.root_fingerprint:
            raise StorageFailure(
                "legacy_fingerprint_mismatch", identifier=authorization.source_alias
            )

        collection_count = 0
        records_seen = 0
        imported_count = 0
        existing_count = 0
        mapping_codes: set[str] = set()
        imported_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        workspace = authorization.restore_workspace.resolve(strict=True)
        with TemporaryDirectory(prefix="legacy-read-", dir=workspace) as temporary:
            staged = Path(temporary) / "source"
            shutil.copytree(source, staged)
            staged_before = legacy_tree_sha256(staged)
            if staged_before != before:
                raise StorageFailure(
                    "legacy_staging_mismatch", identifier=authorization.source_alias
                )
            client = self._client_factory(str(staged))
            try:
                listed: list[Any] = []
                collection_offset = 0
                while True:
                    collection_page = list(
                        client.list_collections(
                            limit=page_size,
                            offset=collection_offset,
                        )
                    )
                    if not collection_page:
                        break
                    listed.extend(collection_page)
                    collection_offset += len(collection_page)
                names = sorted(
                    str(item if isinstance(item, str) else item.name) for item in listed
                )
                collection_count = len(names)
                for collection_name in names:
                    collection = client.get_collection(collection_name)
                    offset = 0
                    while True:
                        page = collection.get(
                            limit=page_size,
                            offset=offset,
                            include=["documents", "metadatas"],
                        )
                        identities = page.get("ids") or []
                        if not identities:
                            break
                        documents = page.get("documents") or []
                        metadatas = page.get("metadatas") or []
                        if not (
                            len(identities) == len(documents) == len(metadatas)
                            and len(identities) <= page_size
                        ):
                            raise StorageFailure("legacy_page_shape_invalid")
                        ordered = sorted(
                            zip(identities, documents, metadatas, strict=True),
                            key=lambda item: str(item[0]),
                        )
                        for identity, document, metadata in ordered:
                            legacy_id = str(identity)
                            if document is None:
                                raise StorageFailure(
                                    "legacy_document_missing", identifier=legacy_id
                                )
                            raw_metadata, normalized_metadata, codes = (
                                self._map_metadata(metadata)
                            )
                            scope_id = self._scope_id(
                                normalized_metadata,
                                authorization.root_fingerprint,
                                codes,
                            )
                            observed_at = self._observed_at(normalized_metadata, codes)
                            codes.add("pair_unproven")
                            mapping_codes.update(codes)
                            content = str(document)
                            inserted = self.repository.import_legacy_fragment(
                                source_alias=authorization.source_alias,
                                root_fingerprint=authorization.root_fingerprint,
                                collection_name=collection_name,
                                legacy_id=legacy_id,
                                scope_id=scope_id,
                                content=content,
                                content_sha256=hashlib.sha256(
                                    content.encode("utf-8")
                                ).hexdigest(),
                                observed_at=observed_at,
                                raw_metadata_json=raw_metadata,
                                status_codes=tuple(sorted(codes)),
                                imported_at=imported_at,
                            )
                            records_seen += 1
                            imported_count += int(inserted)
                            existing_count += int(not inserted)
                        offset += len(identities)
            finally:
                # Chroma 1.5 exposes close() on Client, but omits it from ClientAPI.
                getattr(client, "close")()

        after = legacy_tree_sha256(source)
        if after != before:
            raise StorageFailure(
                "legacy_source_mutated", identifier=authorization.source_alias
            )
        self.repository.record_legacy_import_evidence(
            source_alias=authorization.source_alias,
            root_fingerprint=authorization.root_fingerprint,
            reason_codes=authorization.anomaly_codes,
        )
        return LegacyImportResult(
            source_alias=authorization.source_alias,
            root_fingerprint=authorization.root_fingerprint,
            collection_count=collection_count,
            records_seen=records_seen,
            imported_count=imported_count,
            existing_count=existing_count,
            fragment_count=records_seen,
            source_before_sha256=before,
            source_after_sha256=after,
            mapping_codes=tuple(sorted(mapping_codes)),
            anomaly_counts={code: 1 for code in sorted(set(authorization.anomaly_codes))},
        )

    @staticmethod
    def _validate_authorization(authorization: LegacyImportAuthorization) -> Path:
        if (
            authorization.evidence_status != "pass"
            or len(authorization.evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in authorization.evidence_sha256
            )
        ):
            raise StorageFailure("legacy_restore_evidence_missing")
        if (
            authorization.source_alias,
            authorization.root_fingerprint,
        ) not in set(authorization.allowlisted_sources):
            raise StorageFailure(
                "legacy_source_not_allowlisted", identifier=authorization.source_alias
            )
        try:
            workspace = authorization.restore_workspace.resolve(strict=True)
            supplied_source = authorization.disposable_root
            source = supplied_source.resolve(strict=True)
        except OSError as error:
            raise StorageFailure("legacy_source_missing") from error
        if not workspace.is_dir() or not source.is_dir():
            raise StorageFailure("legacy_source_not_directory")
        if source == workspace or not source.is_relative_to(workspace):
            raise StorageFailure("legacy_source_outside_restore_workspace")
        relative = supplied_source.absolute().relative_to(
            authorization.restore_workspace.absolute()
        )
        cursor = authorization.restore_workspace.absolute()
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise StorageFailure("legacy_source_symlink_rejected")
        for forbidden in authorization.forbidden_roots:
            try:
                denied = forbidden.resolve(strict=True)
            except OSError:
                continue
            if (
                source == denied
                or source.is_relative_to(denied)
                or denied.is_relative_to(source)
            ):
                raise StorageFailure("legacy_source_forbidden")
        return source

    def _map_metadata(
        self, metadata: Any
    ) -> tuple[str, dict[str, Any], set[str]]:
        codes: set[str] = set()
        if not isinstance(metadata, Mapping):
            metadata = {}
            codes.add("metadata_malformed")
        normalized = self._normalize_mapping(metadata, codes)
        if set(normalized) - self._known_metadata:
            codes.add("metadata_unmapped")
        return (
            json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            normalized,
            codes,
        )

    @classmethod
    def _normalize_mapping(
        cls, metadata: Mapping[Any, Any], codes: set[str]
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            name = str(key)
            if isinstance(value, float) and not math.isfinite(value):
                codes.add("metadata_nonfinite")
                normalized[name] = {
                    "__legacy_nonfinite__": "nan" if math.isnan(value) else "infinity"
                }
            elif (
                isinstance(value, str)
                and name in {"confidence", "intensity", "timestamp_unix"}
                and value.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity"}
            ):
                codes.add("metadata_nonfinite")
                normalized[name] = value
            elif isinstance(value, (str, int, float, bool)) or value is None:
                normalized[name] = value
            else:
                codes.add("metadata_malformed")
                normalized[name] = {
                    "__legacy_repr_sha256__": hashlib.sha256(
                        repr(value).encode("utf-8")
                    ).hexdigest()
                }
        return normalized

    @staticmethod
    def _scope_id(
        metadata: Mapping[str, Any], root_fingerprint: str, codes: set[str]
    ) -> str:
        supplied = metadata.get("user_id")
        if isinstance(supplied, str) and supplied.strip():
            return supplied.strip()
        codes.add("scope_missing")
        return f"legacy-unscoped:{root_fingerprint[:16]}"

    @staticmethod
    def _observed_at(
        metadata: Mapping[str, Any], codes: set[str]
    ) -> str | None:
        supplied = metadata.get("timestamp")
        if supplied is None or supplied == "":
            codes.add("timestamp_missing")
            return None
        if not isinstance(supplied, str):
            codes.add("timestamp_malformed")
            return None
        try:
            datetime.fromisoformat(supplied.replace("Z", "+00:00"))
        except ValueError:
            codes.add("timestamp_malformed")
            return None
        return supplied
