"""Fail-closed contracts for Aura's sanitized memory benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

_MANIFEST_KEYS = {
    "admission_thresholds",
    "arms",
    "benchmark_code_commit",
    "candidate_caps",
    "case_ids",
    "controls",
    "corpus_sha256",
    "corpus_version",
    "embedding_fingerprint_slot",
    "expected_origins",
    "fts_tokenizer",
    "gates",
    "hardware_label",
    "load_event_count",
    "load_seed",
    "max_corpus_bytes",
    "max_records",
    "page_bounds",
    "rrf_k",
    "runtime_label",
    "schema_version",
    "timing_repetitions",
    "vector_metric",
}
_REQUIRED_CASE_CLASSES = {
    "absent_fact",
    "correction",
    "cross_scope_collision",
    "direct_recall",
    "distractor",
    "duplicate",
    "pagination",
    "paraphrased_recall",
    "prompt_injection",
    "provenance",
    "temporal",
}
_REQUIRED_CONTROLS = {
    "absent_fact_abstains",
    "correction_excludes_stale",
    "faithful_exact_fact_retrieves",
    "provenance_drop_detected",
    "scope_leak_detected",
    "stale_preference_detected",
}
_PRIVACY_MARKERS = (
    b"/home/ty/",
    b"aura_chroma_db",
    b"aura_data/",
    b"TY_PRIVATE_SENTINEL",
)
_ALLOWED_SCOPES = {"scope-synthetic-a", "scope-synthetic-b"}


class InstrumentValidationError(ValueError):
    """Content-free benchmark validation error with a stable code."""

    def __init__(self, code: str, *, identifier: str | None = None) -> None:
        self.code = code
        self.identifier = identifier
        detail = f" identifier={identifier}" if identifier is not None else ""
        super().__init__(f"benchmark instrument invalid: code={code}{detail}")


class OutcomeStatus(str, Enum):
    """Truthful disposition of a benchmark or adoption gate."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class BenchmarkAbort(RuntimeError):
    """Typed resource/timeout termination raised by an injected retriever."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"benchmark aborted: code={code}")


@dataclass(frozen=True, slots=True)
class LoadedInstrument:
    """Validated immutable benchmark inputs."""

    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    case_ids: tuple[str, ...]
    load_event_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """Content-free query identity passed to an injected retriever."""

    case_id: str
    case_class: str
    scope_id: str
    expected_origin_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    """Candidate facts needed to evaluate scope, provenance, and freshness."""

    origin_id: str
    scope_id: str
    provenance_event_ids: tuple[str, ...]
    active: bool
    content_hash: str


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """One bounded retrieval observation."""

    candidates: tuple[RetrievedCandidate, ...]
    elapsed_ms: float
    complete: bool = True
    condition_code: str | None = None


class BenchmarkRetriever(Protocol):
    """Minimal injected retrieval boundary used by the instrument."""

    controls: frozenset[str]

    def retrieve(
        self, query: BenchmarkQuery, *, repetition: int
    ) -> RetrievalResponse:
        """Return at most five ranked candidates for one fixed case."""
        ...


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Pre-registered aggregate measurements only."""

    direct_paraphrase_recall_at_5: float
    explicit_update_accuracy: float
    critical_absent_abstention: float
    cross_scope_leaks: int
    selected_without_provenance: int
    stale_selected: int
    duplicate_selected: int
    warmed_local_p95_ms: float
    storage_bytes: int
    restore_pass: bool


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    """Content-free evidence for one case/repetition."""

    case_id: str
    repetition: int
    selected_origin_ids: tuple[str, ...]
    elapsed_ms: float
    codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Content-free report for one neutral benchmark arm."""

    status: OutcomeStatus
    code: str
    arm_name: str
    salience_contribution: float
    corpus_sha256: str
    metrics: BenchmarkMetrics | None
    evidence: tuple[CaseEvidence, ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return only IDs, counts, timings, codes, and the corpus hash."""
        raise BenchmarkAbort("not_implemented")


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Outcome of a pre-registered alternative-adoption decision."""

    status: OutcomeStatus
    code: str


def run_benchmark(
    instrument: LoadedInstrument,
    retriever: BenchmarkRetriever,
    *,
    arm_name: str = "neutral",
) -> BenchmarkReport:
    """Evaluate a retriever against fixed cases and resource semantics."""
    raise BenchmarkAbort("not_implemented")


def evaluate_alternative(
    baseline: BenchmarkReport,
    alternative: BenchmarkReport,
    *,
    cycles_used: int,
    elapsed_work_seconds: int = 0,
) -> GateOutcome:
    """Apply the fixed improvement and bounded-cycle adoption rule."""
    raise BenchmarkAbort("not_implemented")


def load_instrument(corpus_path: Path, manifest_path: Path) -> LoadedInstrument:
    """Load and validate a bounded corpus plus its frozen manifest."""
    manifest = _load_manifest(manifest_path)
    corpus = corpus_path.read_bytes()
    if not corpus:
        raise InstrumentValidationError("corpus_empty")
    if len(corpus) > manifest["max_corpus_bytes"]:
        raise InstrumentValidationError("corpus_too_large")
    if not corpus.endswith(b"\n"):
        raise InstrumentValidationError("corpus_truncated")
    if any(marker.lower() in corpus.lower() for marker in _PRIVACY_MARKERS):
        raise InstrumentValidationError("privacy_marker_detected")
    if hashlib.sha256(corpus).hexdigest() != manifest["corpus_sha256"]:
        raise InstrumentValidationError("corpus_digest_mismatch")

    records = _load_records(corpus, max_records=manifest["max_records"])
    case_ids = _validate_inventory(records, manifest)
    load_events = _validate_load_slice(records, manifest)
    return LoadedInstrument(
        manifest=manifest,
        records=records,
        case_ids=case_ids,
        load_event_count=len(load_events),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    """Parse the small manifest and enforce every pre-registered field."""
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > 256_000:
            raise InstrumentValidationError("manifest_contract_invalid")
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentValidationError("manifest_contract_invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise InstrumentValidationError("manifest_contract_invalid")

    case_ids = manifest["case_ids"]
    expected_origins = manifest["expected_origins"]
    admission = manifest["admission_thresholds"]
    contract_is_valid = (
        manifest["schema_version"] == 1
        and manifest["corpus_version"] == "memory-eval-v1"
        and isinstance(manifest["corpus_sha256"], str)
        and len(manifest["corpus_sha256"]) == 64
        and isinstance(case_ids, list)
        and all(isinstance(case_id, str) and case_id for case_id in case_ids)
        and len(set(case_ids)) == len(case_ids)
        and isinstance(expected_origins, dict)
        and manifest["fts_tokenizer"] == "unicode61 remove_diacritics 2"
        and manifest["vector_metric"] == "cosine"
        and isinstance(manifest["embedding_fingerprint_slot"], str)
        and bool(manifest["embedding_fingerprint_slot"])
        and manifest["candidate_caps"] == {"lexical": 50, "vector": 50}
        and manifest["rrf_k"] == 60
        and admission
        == {
            "lexical_min_token_coverage": 0.5,
            "vector_min_cosine_similarity": 0.35,
        }
        and manifest["page_bounds"] == {"default": 20, "maximum": 100}
        and isinstance(manifest["runtime_label"], str)
        and bool(manifest["runtime_label"])
        and isinstance(manifest["hardware_label"], str)
        and bool(manifest["hardware_label"])
        and manifest["timing_repetitions"] == 3
        and manifest["load_seed"] == 3202
        and manifest["load_event_count"] == 10_000
        and isinstance(manifest["benchmark_code_commit"], str)
        and bool(manifest["benchmark_code_commit"])
        and isinstance(manifest["max_corpus_bytes"], int)
        and manifest["max_corpus_bytes"] > 0
        and isinstance(manifest["max_records"], int)
        and 10_000 <= manifest["max_records"] <= 20_000
        and set(manifest["controls"]) == _REQUIRED_CONTROLS
        and len(manifest["controls"]) == len(_REQUIRED_CONTROLS)
        and manifest["arms"]
        == {
            "constant_salience_control": {"salience_contribution": 0.0},
            "neutral": {"salience_contribution": 0.0},
        }
        and manifest["gates"]
        == {
            "critical_absent_abstention": 1.0,
            "cross_scope_leaks": 0,
            "direct_paraphrase_recall_at_5": 0.9,
            "explicit_update_accuracy": 0.9,
            "max_warmed_local_p95_ms": 250.0,
            "selected_without_provenance": 0,
        }
    )
    if not contract_is_valid:
        raise InstrumentValidationError("manifest_contract_invalid")
    if set(case_ids) != set(expected_origins):
        raise InstrumentValidationError("case_inventory_mismatch")
    return manifest


def _load_records(corpus: bytes, *, max_records: int) -> tuple[dict[str, Any], ...]:
    """Parse newline-delimited records without ever echoing their content."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(corpus.splitlines(), start=1):
        if not line:
            raise InstrumentValidationError(
                "corpus_malformed", identifier=f"line-{line_number}"
            )
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InstrumentValidationError(
                "corpus_malformed", identifier=f"line-{line_number}"
            ) from error
        if not isinstance(record, dict):
            raise InstrumentValidationError(
                "corpus_malformed", identifier=f"line-{line_number}"
            )
        records.append(record)
        if len(records) > max_records:
            raise InstrumentValidationError("corpus_record_limit")
    return tuple(records)


def _validate_inventory(
    records: tuple[dict[str, Any], ...], manifest: dict[str, Any]
) -> tuple[str, ...]:
    """Require one known query per case and safe, typed fixture records."""
    query_cases: dict[str, dict[str, Any]] = {}
    origin_ids: set[str] = set()
    allowed_cases = set(manifest["case_ids"]) | {"load_slice"}
    for line_number, record in enumerate(records, start=1):
        case_id = record.get("case_id")
        record_type = record.get("record_type")
        scope_id = record.get("scope_id")
        identifier = case_id if isinstance(case_id, str) else f"line-{line_number}"
        if (
            not isinstance(case_id, str)
            or case_id not in allowed_cases
            or record_type not in {"event", "memory", "query"}
            or scope_id not in _ALLOWED_SCOPES
        ):
            raise InstrumentValidationError("corpus_record_invalid", identifier=identifier)
        if record_type == "query":
            if case_id in query_cases:
                raise InstrumentValidationError("case_duplicate", identifier=case_id)
            expected = record.get("expected_origin_ids")
            if (
                not isinstance(record.get("query"), str)
                or not isinstance(expected, list)
                or expected != manifest["expected_origins"].get(case_id)
            ):
                raise InstrumentValidationError(
                    "case_expectation_invalid", identifier=case_id
                )
            query_cases[case_id] = record
            continue

        origin_id = record.get("origin_id")
        if not isinstance(origin_id, str) or not origin_id or origin_id in origin_ids:
            raise InstrumentValidationError("origin_invalid", identifier=identifier)
        if not isinstance(record.get("text"), str) or not record["text"]:
            raise InstrumentValidationError("corpus_record_invalid", identifier=identifier)
        origin_ids.add(origin_id)
        if record_type == "memory":
            provenance = record.get("provenance_event_ids")
            if not isinstance(provenance, list) or not provenance:
                raise InstrumentValidationError("provenance_missing", identifier=origin_id)

    if tuple(sorted(query_cases)) != tuple(sorted(manifest["case_ids"])):
        raise InstrumentValidationError("case_inventory_mismatch")
    if {case["case_class"] for case in query_cases.values()} != _REQUIRED_CASE_CLASSES:
        raise InstrumentValidationError("case_inventory_mismatch")
    missing_origins = {
        origin_id
        for expected in manifest["expected_origins"].values()
        for origin_id in expected
        if origin_id not in origin_ids
    }
    if missing_origins:
        raise InstrumentValidationError("expected_origin_missing")
    return tuple(sorted(query_cases))


def _validate_load_slice(
    records: tuple[dict[str, Any], ...], manifest: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Prove the 10k-event load slice is materialized and deterministic."""
    load_events = tuple(
        record
        for record in records
        if record["record_type"] == "event" and record["case_id"] == "load_slice"
    )
    if len(load_events) != manifest["load_event_count"]:
        raise InstrumentValidationError("load_slice_incomplete")
    for index, record in enumerate(load_events):
        if (
            record.get("origin_id") != f"evt-load-{index:05d}"
            or record.get("sequence_index") != index
            or record.get("seed") != manifest["load_seed"]
        ):
            raise InstrumentValidationError(
                "load_slice_nondeterministic", identifier=f"load-{index:05d}"
            )
    return load_events
