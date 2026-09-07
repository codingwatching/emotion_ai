"""Verify the smoke instrument cannot pass through abstention or missing coverage."""

import hashlib
import json
from pathlib import Path

from aura_backend.conversation.emotion_assessment import EmotionAssessment
from aura_backend.conversation.evaluate_emotion import CASES, score_results


def test_recorded_live_evidence_matches_sources_and_recomputed_score() -> None:
    """Audit the saved result without treating historical evidence as a fresh run."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/evidence/2026-09-07-emotion-smoke.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = {case.case_id: case for case in CASES}
    results = {}
    assert len(report["rows"]) == len(cases)
    assert {row["case_id"] for row in report["rows"]} == set(cases)
    for row in report["rows"]:
        case = cases[row["case_id"]]
        assessment = row["assessment"]
        assert (
            assessment["source_sha256"]
            == hashlib.sha256(case.source.encode()).hexdigest()
        )
        assert all(quote in case.source for quote in assessment["evidence"])
        results[case.case_id] = EmotionAssessment(
            **{
                **assessment,
                "evidence": tuple(assessment["evidence"]),
            }
        )
    recomputed = json.loads(json.dumps(score_results(results)))
    assert all(report[key] == value for key, value in recomputed.items())


def test_empty_results_do_not_pass_or_count_as_correct() -> None:
    result = score_results({})
    assert result["complete"] is False
    assert result["passed"] is False
    assert result["correct"] == 0


def test_always_abstaining_fails_the_positive_controls() -> None:
    result = score_results(
        {
            case.case_id: EmotionAssessment(
                subject="user", status="abstained", source_sha256="a" * 64
            )
            for case in CASES
        }
    )
    assert result["complete"] is True
    assert result["passed"] is False
    assert result["correct"] == result["negative_total"] == 7


def test_invalid_or_unavailable_is_not_correct_abstention() -> None:
    for status in ("invalid", "unavailable"):
        result = score_results(
            {
                case.case_id: EmotionAssessment(
                    subject="user", status=status, source_sha256="a" * 64
                )
                for case in CASES
            }
        )
        assert result["complete"] is True
        assert result["passed"] is False
        assert result["correct"] == 0


def test_confidently_labeling_everything_counts_false_positives() -> None:
    result = score_results(
        {
            case.case_id: EmotionAssessment(
                subject="user",
                status="inferred",
                source_sha256="a" * 64,
                emotion="Happy",
            )
            for case in CASES
        }
    )
    assert result["negative_false_positives"] == 7
    assert result["passed"] is False
