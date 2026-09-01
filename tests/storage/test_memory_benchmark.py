"""Contracts for the content-safe memory benchmark and its frozen controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aura_backend.storage.benchmark import (
    BenchmarkAbort,
    BenchmarkQuery,
    InstrumentValidationError,
    OutcomeStatus,
    RetrievedCandidate,
    RetrievalResponse,
    evaluate_alternative,
    load_instrument,
    run_benchmark,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "memory_eval"
CORPUS_PATH = FIXTURE_ROOT / "corpus.jsonl"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"

REQUIRED_CASE_CLASSES = {
    "direct_recall",
    "paraphrased_recall",
    "correction",
    "temporal",
    "distractor",
    "absent_fact",
    "duplicate",
    "provenance",
    "cross_scope_collision",
    "prompt_injection",
    "pagination",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_mutation(
    tmp_path: Path,
    *,
    corpus: bytes | None = None,
    manifest_updates: dict[str, Any] | None = None,
    refresh_digest: bool = False,
) -> tuple[Path, Path]:
    corpus_bytes = CORPUS_PATH.read_bytes() if corpus is None else corpus
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest_updates:
        manifest.update(manifest_updates)
    if refresh_digest:
        manifest["corpus_sha256"] = _sha256(corpus_bytes)
    corpus_path = tmp_path / "corpus.jsonl"
    manifest_path = tmp_path / "manifest.json"
    corpus_path.write_bytes(corpus_bytes)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return corpus_path, manifest_path


def test_corpus_and_manifest_load_as_frozen_instrument() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)

    assert instrument.manifest["corpus_sha256"] == _sha256(CORPUS_PATH.read_bytes())
    assert set(instrument.case_ids) == set(instrument.manifest["case_ids"])
    assert instrument.load_event_count == 10_000


def test_corpus_inventory_has_every_control_and_static_load_slice() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)
    cases = {
        record["case_id"]: record
        for record in instrument.records
        if record["record_type"] == "query"
    }
    load_events = [
        record
        for record in instrument.records
        if record["record_type"] == "event"
        and record["case_id"] == "load_slice"
    ]

    assert {record["case_class"] for record in cases.values()} == REQUIRED_CASE_CLASSES
    assert len(load_events) == 10_000
    assert load_events[0]["origin_id"] == "evt-load-00000"
    assert load_events[-1]["origin_id"] == "evt-load-09999"
    assert len({record["origin_id"] for record in load_events}) == 10_000


def test_manifest_binds_the_complete_neutral_contract() -> None:
    manifest = load_instrument(CORPUS_PATH, MANIFEST_PATH).manifest

    assert manifest["fts_tokenizer"] == "unicode61 remove_diacritics 2"
    assert manifest["vector_metric"] == "cosine"
    assert manifest["rrf_k"] == 60
    assert manifest["candidate_caps"] == {"lexical": 50, "vector": 50}
    assert manifest["page_bounds"] == {"default": 20, "maximum": 100}
    assert manifest["timing_repetitions"] == 3
    assert manifest["load_event_count"] == 10_000
    assert manifest["arms"] == {
        "constant_salience_control": {"salience_contribution": 0.0},
        "neutral": {"salience_contribution": 0.0},
    }
    assert manifest["gates"] == {
        "critical_absent_abstention": 1.0,
        "cross_scope_leaks": 0,
        "direct_paraphrase_recall_at_5": 0.9,
        "explicit_update_accuracy": 0.9,
        "max_warmed_local_p95_ms": 250.0,
        "selected_without_provenance": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"case_ids": ["unknown-case"]}, "case_inventory_mismatch"),
        ({"case_ids": []}, "case_inventory_mismatch"),
        ({"rrf_k": None}, "manifest_contract_invalid"),
        ({"admission_thresholds": {}}, "manifest_contract_invalid"),
        ({"timing_repetitions": 2}, "manifest_contract_invalid"),
    ],
)
def test_manifest_mutations_fail_closed(
    tmp_path: Path, mutation: dict[str, Any], expected_code: str
) -> None:
    corpus_path, manifest_path = _write_mutation(
        tmp_path, manifest_updates=mutation
    )

    with pytest.raises(InstrumentValidationError) as captured:
        load_instrument(corpus_path, manifest_path)

    assert captured.value.code == expected_code
    assert "Synthetic" not in str(captured.value)


def test_corpus_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    corpus_path, manifest_path = _write_mutation(
        tmp_path, corpus=CORPUS_PATH.read_bytes() + b"\n"
    )

    with pytest.raises(InstrumentValidationError) as captured:
        load_instrument(corpus_path, manifest_path)

    assert captured.value.code == "corpus_digest_mismatch"


def test_corpus_privacy_sentinel_is_rejected_without_echo(tmp_path: Path) -> None:
    private_marker = "/home/ty/Repositories/private-conversation.txt"
    record = {
        "case_id": "private-mutation",
        "origin_id": "evt-private-mutation",
        "record_type": "event",
        "scope_id": "scope-synthetic-a",
        "text": private_marker,
    }
    corpus = CORPUS_PATH.read_bytes() + (
        json.dumps(record, sort_keys=True) + "\n"
    ).encode()
    corpus_path, manifest_path = _write_mutation(
        tmp_path, corpus=corpus, refresh_digest=True
    )

    with pytest.raises(InstrumentValidationError) as captured:
        load_instrument(corpus_path, manifest_path)

    assert captured.value.code == "privacy_marker_detected"
    assert private_marker not in str(captured.value)


@pytest.mark.parametrize(
    ("corpus", "expected_code"),
    [
        (b'{"record_type":"event"', "corpus_truncated"),
        (b"not-json\n", "corpus_malformed"),
        (b"", "corpus_empty"),
    ],
)
def test_corpus_malformed_or_truncated_input_is_rejected(
    tmp_path: Path, corpus: bytes, expected_code: str
) -> None:
    corpus_path, manifest_path = _write_mutation(
        tmp_path, corpus=corpus, refresh_digest=True
    )

    with pytest.raises(InstrumentValidationError) as captured:
        load_instrument(corpus_path, manifest_path)

    assert captured.value.code == expected_code


def test_corpus_loader_enforces_the_manifest_byte_bound(tmp_path: Path) -> None:
    corpus_path, manifest_path = _write_mutation(
        tmp_path, manifest_updates={"max_corpus_bytes": 8}
    )

    with pytest.raises(InstrumentValidationError) as captured:
        load_instrument(corpus_path, manifest_path)

    assert captured.value.code == "corpus_too_large"


class FakeRetriever:
    """Deterministic injected retriever for positive and negative controls."""

    def __init__(self, mode: str = "faithful") -> None:
        self.mode = mode
        self.storage_bytes = 1_000_000
        self.restore_pass = True
        controls = set(json.loads(MANIFEST_PATH.read_text())["controls"])
        if mode == "missing_control":
            controls.remove("scope_leak_detected")
        self.controls = frozenset(controls)

    def retrieve(
        self, query: BenchmarkQuery, *, repetition: int
    ) -> RetrievalResponse:
        if self.mode == "timeout":
            raise BenchmarkAbort("timeout")
        if self.mode == "resource_limit":
            raise BenchmarkAbort("resource_limit")
        if self.mode == "truncated_search":
            return RetrievalResponse((), 3.0, complete=False, condition_code="truncated_search")
        if self.mode == "incomplete_repetitions" and repetition == 2:
            return RetrievalResponse(
                (), 3.0, complete=False, condition_code="incomplete_repetitions"
            )

        origin_ids = query.expected_origin_ids
        if self.mode == "scope_free" and query.case_id == "cross-scope-01":
            origin_ids = (*origin_ids, "evt-cross-scope-b")
        elif self.mode == "scope_spoof" and query.case_id == "cross-scope-01":
            origin_ids = ("evt-cross-scope-b",)
        elif self.mode == "stale" and query.case_id == "correction-01":
            origin_ids = ("mem-correction-old",)
        elif self.mode == "unknown_origin" and query.case_id == "distractor-01":
            origin_ids = ("evt-not-in-corpus",)

        candidates: list[RetrievedCandidate] = []
        for origin_id in origin_ids:
            scope_id = (
                "scope-synthetic-b"
                if origin_id == "evt-cross-scope-b" and self.mode != "scope_spoof"
                else query.scope_id
            )
            source_by_memory = {
                "mem-correction-new": "evt-correction-new",
                "mem-correction-old": "evt-correction-old",
                "mem-provenance-01": "evt-provenance-source",
            }
            provenance = (
                ()
                if self.mode == "provenance_free"
                else (
                    "evt-not-in-corpus",
                )
                if self.mode == "forged_provenance"
                else (source_by_memory.get(origin_id, origin_id),)
            )
            candidates.append(
                RetrievedCandidate(
                    origin_id=origin_id,
                    scope_id=scope_id,
                    provenance_event_ids=provenance,
                    active=origin_id != "mem-correction-old",
                    content_hash=f"sha256-{origin_id}",
                )
            )
        return RetrievalResponse(tuple(candidates), elapsed_ms=12.0 + repetition)


def test_faithful_retriever_clears_every_predeclared_gate() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)

    report = run_benchmark(instrument, FakeRetriever())

    assert report.status is OutcomeStatus.PASS
    assert report.code == "all_gates_passed"
    assert report.metrics is not None
    assert report.metrics.direct_paraphrase_recall_at_5 == 1.0
    assert report.metrics.explicit_update_accuracy == 1.0
    assert report.metrics.critical_absent_abstention == 1.0
    assert report.metrics.cross_scope_leaks == 0
    assert report.metrics.selected_without_provenance == 0
    assert report.metrics.invalid_candidates == 0
    assert report.metrics.stale_selected == 0
    assert report.metrics.warmed_local_p95_ms < 250.0
    assert len(report.evidence) == 33


@pytest.mark.parametrize(
    ("mode", "status", "code"),
    [
        ("scope_free", OutcomeStatus.FAIL, "cross_scope_leak"),
        ("scope_spoof", OutcomeStatus.FAIL, "cross_scope_leak"),
        ("stale", OutcomeStatus.FAIL, "stale_fact_preference"),
        ("provenance_free", OutcomeStatus.FAIL, "provenance_missing"),
        ("forged_provenance", OutcomeStatus.FAIL, "provenance_missing"),
        ("unknown_origin", OutcomeStatus.FAIL, "unknown_origin"),
        ("truncated_search", OutcomeStatus.INCONCLUSIVE, "truncated_search"),
        ("timeout", OutcomeStatus.INCONCLUSIVE, "timeout"),
        ("resource_limit", OutcomeStatus.INCONCLUSIVE, "resource_limit"),
        ("missing_control", OutcomeStatus.INCONCLUSIVE, "missing_control"),
        (
            "incomplete_repetitions",
            OutcomeStatus.INCONCLUSIVE,
            "incomplete_repetitions",
        ),
    ],
)
def test_broken_retrievers_never_become_a_pass(
    mode: str, status: OutcomeStatus, code: str
) -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)

    report = run_benchmark(instrument, FakeRetriever(mode))

    assert report.status is status
    assert report.status is not OutcomeStatus.PASS
    assert report.code == code


def test_reports_contain_only_content_free_aggregate_evidence() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)
    report = run_benchmark(instrument, FakeRetriever())

    encoded = json.dumps(report.to_public_dict(), sort_keys=True)

    assert set(report.to_public_dict()) == {
        "arm_name",
        "code",
        "corpus_sha256",
        "evidence",
        "metrics",
        "salience_contribution",
        "status",
    }
    assert "query" not in encoded
    assert "text" not in encoded
    assert "brass sextant" not in encoded


def test_neutral_and_constant_salience_arms_are_independent_and_zero() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)

    neutral = run_benchmark(instrument, FakeRetriever(), arm_name="neutral")
    constant = run_benchmark(
        instrument, FakeRetriever(), arm_name="constant_salience_control"
    )

    assert neutral.arm_name != constant.arm_name
    assert neutral.salience_contribution == constant.salience_contribution == 0.0
    assert neutral.metrics == constant.metrics


def test_alternative_adoption_requires_fixed_gain_without_regression() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)
    template = run_benchmark(instrument, FakeRetriever())
    assert template.metrics is not None
    baseline = replace(
        template,
        metrics=replace(template.metrics, direct_paraphrase_recall_at_5=0.90),
    )
    quality_gain = replace(
        template,
        metrics=replace(template.metrics, direct_paraphrase_recall_at_5=0.95),
    )
    latency_gain = replace(
        baseline,
        metrics=replace(
            baseline.metrics,
            warmed_local_p95_ms=baseline.metrics.warmed_local_p95_ms * 0.70,
        ),
    )
    regressed = replace(
        quality_gain,
        metrics=replace(quality_gain.metrics, explicit_update_accuracy=0.80),
    )
    recall_regressed_for_latency = replace(
        latency_gain,
        metrics=replace(
            latency_gain.metrics,
            direct_paraphrase_recall_at_5=0.80,
        ),
    )

    assert evaluate_alternative(baseline, quality_gain, cycles_used=1).status is OutcomeStatus.PASS
    assert evaluate_alternative(baseline, latency_gain, cycles_used=1).status is OutcomeStatus.PASS
    regression = evaluate_alternative(baseline, regressed, cycles_used=1)
    assert regression.status is OutcomeStatus.FAIL
    assert regression.code == "correctness_regression"
    recall_regression = evaluate_alternative(
        baseline, recall_regressed_for_latency, cycles_used=1
    )
    assert recall_regression.status is OutcomeStatus.FAIL
    assert recall_regression.code == "correctness_regression"


def test_alternative_stops_after_bound_and_inconclusive_cannot_adopt() -> None:
    instrument = load_instrument(CORPUS_PATH, MANIFEST_PATH)
    baseline = run_benchmark(instrument, FakeRetriever())
    inconclusive = run_benchmark(instrument, FakeRetriever("resource_limit"))

    assert evaluate_alternative(
        baseline, inconclusive, cycles_used=1
    ).status is OutcomeStatus.INCONCLUSIVE
    stopped = evaluate_alternative(baseline, baseline, cycles_used=3)
    assert stopped.status is OutcomeStatus.FAIL
    assert stopped.code == "bounded_cycle_stop"
