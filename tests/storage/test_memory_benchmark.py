"""Contracts for the content-safe memory benchmark and its frozen controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from aura_backend.storage.benchmark import InstrumentValidationError, load_instrument

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
