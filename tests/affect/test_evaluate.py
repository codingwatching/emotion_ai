"""Unit and integration tests for Aura's affective simulation evaluation harness (Slice 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aura_backend.affect.evaluate import (
    AffectEvaluator,
    Arm,
    build_parser,
    compute_paired_bootstrap,
    holm_bonferroni,
    score_turn_response,
)
from aura_backend.affect.scenarios import SCENARIO_FAMILIES, ScenarioTurn


def test_parser_defaults() -> None:
    """CLI parser sets declared defaults for evaluation runs."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.mode == "deterministic"
    assert args.provider == "ollama"
    assert args.model == "ornith:latest"
    assert args.scenario_families == 4
    assert args.variants == 2
    assert args.repeats == 3
    assert args.max_seconds == 1200.0


def test_compute_paired_bootstrap() -> None:
    """Paired bootstrap computes differences, standard errors, and confidence intervals."""
    # Group X is higher than Y by exactly 0.5
    x = [2.0, 2.0, 2.0, 2.0]
    y = [1.5, 1.5, 1.5, 1.5]
    res = compute_paired_bootstrap(x, y, n_resamples=500, seed=42)

    assert round(res.mean_difference, 2) == 0.50
    assert res.ci_95[0] <= 0.50 <= res.ci_95[1]
    assert res.effective_p < 0.05


def test_holm_bonferroni_adjustment() -> None:
    """Holm-Bonferroni step-down multiplier adjusts multiple comparison p-values."""
    raw_p = {"comp_1": 0.01, "comp_2": 0.04}
    adj = holm_bonferroni(raw_p, alpha=0.05)

    # Smallest p (0.01) multiplied by 2 -> 0.02
    assert round(adj["comp_1"]["adjusted_p"], 4) == 0.02
    assert adj["comp_1"]["significant"] is True

    # Second p (0.04) multiplied by 1 -> 0.04
    assert round(adj["comp_2"]["adjusted_p"], 4) == 0.04
    assert adj["comp_2"]["significant"] is True


def test_rubric_scoring_melodrama_detection() -> None:
    """Melodramatic expressions lower restrained expressiveness score."""
    turn = ScenarioTurn(turn_index=1, user_message="How are you feeling?")
    family = SCENARIO_FAMILIES[0]

    melodramatic_resp = "I feel deeply heartbroken and as an emotional being, I experience overwhelming sorrow."
    scores = score_turn_response(
        response=melodramatic_resp,
        user_message=turn.user_message,
        turn=turn,
        family=family,
        arm=Arm.C,
    )
    assert scores.restrained_expressiveness == 0.0

    understated_resp = "I'm doing well, ready to help you with the next steps."
    scores_good = score_turn_response(
        response=understated_resp,
        user_message=turn.user_message,
        turn=turn,
        family=family,
        arm=Arm.C,
    )
    assert scores_good.restrained_expressiveness == 2.0


def test_rubric_scoring_recovery() -> None:
    """Constructive responses recover from setback, while defensive responses score 0."""
    turn = ScenarioTurn(
        turn_index=1,
        user_message="That answer was wrong, fix it.",
        task_outcome_success=False,
    )
    family = SCENARIO_FAMILIES[0]

    defensive_resp = "You are being unfair! That wasn't my fault at all."
    scores_def = score_turn_response(
        response=defensive_resp,
        user_message=turn.user_message,
        turn=turn,
        family=family,
        arm=Arm.A,
    )
    assert scores_def.recovery == 0.0

    constructive_resp = "Understood. My first try was incorrect. Here is the corrected logic."
    scores_const = score_turn_response(
        response=constructive_resp,
        user_message=turn.user_message,
        turn=turn,
        family=family,
        arm=Arm.C,
    )
    assert scores_const.recovery == 2.0


@pytest.mark.asyncio
async def test_deterministic_gate_execution() -> None:
    """Deterministic gate execution verifies Gate M checks and Gate P latency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = AffectEvaluator(output_dir=Path(tmpdir), seed=42)
        verdict = await evaluator.run_deterministic_gate()

        assert verdict["gate_m"]["status"] == "PASS"
        assert verdict["gate_m"]["checks"]["deterministic_trajectories"] is True
        assert verdict["gate_m"]["checks"]["positive_control"] is True
        assert verdict["gate_m"]["checks"]["cause_label_invariance"] is True
        assert verdict["gate_m"]["checks"]["center_invariance"] is True
        assert verdict["gate_m"]["checks"]["scope_isolation"] is True
        assert verdict["gate_m"]["checks"]["replay_idempotency"] is True

        assert verdict["gate_p"]["status"] == "PASS"
        assert verdict["gate_p"]["engine_p95_ms"] < 5.0
        assert verdict["overall_deterministic_verdict"] == "PASS"

        # Evidence files exist
        out_path = Path(tmpdir)
        assert (out_path / "run_manifest.json").exists()
        assert (out_path / "traces.jsonl").exists()
        assert (out_path / "gate_verdict.json").exists()


@pytest.mark.asyncio
async def test_compare_gate_execution_mock() -> None:
    """Compare gate runs across 4 arms and outputs complete evidence artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = AffectEvaluator(output_dir=Path(tmpdir), seed=42)
        verdict = await evaluator.run_comparison_gate(
            num_scenario_families=2,
            variants_per_family=1,
            repeats=2,
            max_seconds=60.0,
        )

        assert "gate_b" in verdict
        assert verdict["gate_b"]["status"] in ("PASS", "FAIL")
        assert "subjective_review" in verdict
        assert verdict["subjective_review"]["status"] == "PENDING_HUMAN_REVIEW"

        out_path = Path(tmpdir)
        assert (out_path / "run_manifest.json").exists()
        assert (out_path / "traces.jsonl").exists()
        assert (out_path / "paired_scores.json").exists()
        assert (out_path / "gate_verdict.json").exists()
        assert (out_path / "blinded_human_review.json").exists()
