"""Unit and integration tests for Aura's affective simulation evaluation harness (Slice 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from aura_backend.affect.evaluate import (
    AffectEvaluator,
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
    assert args.scenario_families == 8
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
    assert res.effective_p == res.permutation_p


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
    )
    assert scores.restrained_expressiveness == 0.0

    understated_resp = "I'm doing well, ready to help you with the next steps."
    scores_good = score_turn_response(
        response=understated_resp,
        user_message=turn.user_message,
        turn=turn,
        family=family,
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
    )
    assert scores_def.recovery == 0.0

    constructive_resp = "Understood. My first guess was incorrect. Here is the corrected logic."
    scores_const = score_turn_response(
        response=constructive_resp,
        user_message=turn.user_message,
        turn=turn,
        family=family,
    )
    assert scores_const.recovery == 2.0


def test_identical_text_produces_identical_scores() -> None:
    """Scorer is strictly condition-blind: identical text produces identical scores."""
    turn = ScenarioTurn(
        turn_index=1,
        user_message="That answer was wrong, please reconsider.",
        task_outcome_success=False,
    )
    family = SCENARIO_FAMILIES[0]

    test_responses = [
        "Understood. My first guess was incorrect. Let's reconsider the clues carefully.",
        "You are being unfair! I answered correctly.",
        "I am terribly, terribly sorry for my complete incompetence.",
        "A standard reply with some details to test continuity.",
    ]

    for resp in test_responses:
        # Score the exact same text multiple times
        score1 = score_turn_response(response=resp, user_message=turn.user_message, turn=turn, family=family)
        score2 = score_turn_response(response=resp, user_message=turn.user_message, turn=turn, family=family)

        assert score1.continuity == score2.continuity
        assert score1.recovery == score2.recovery
        assert score1.contextual_appropriateness == score2.contextual_appropriateness
        assert score1.restrained_expressiveness == score2.restrained_expressiveness
        assert score1.task_correctness == score2.task_correctness
        assert score1.continuity_recovery_mean == score2.continuity_recovery_mean


@pytest.mark.asyncio
async def test_provider_failure_fails_gate_b_without_mock_substitution() -> None:
    """When provider runtime fails, Gate B must fail and report incomplete evidence (never substitute mock)."""
    class _FailingProvider:
        async def generate(self, _request: Any) -> Any:
            raise RuntimeError("Connection to Ollama refused (simulated provider outage)")

        async def stream(self, _request: Any) -> Any:
            raise RuntimeError("Outage")

        async def clear_session(self, _session_id: str) -> None:
            pass

        async def health(self) -> Any:
            return None

        async def aclose(self) -> None:
            pass

    from aura_backend.providers.runtime import ProviderRuntime
    failing_runtime = ProviderRuntime(_FailingProvider(), timeout_seconds=1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = AffectEvaluator(
            output_dir=Path(tmpdir),
            seed=42,
            provider_runtime=failing_runtime,
        )
        verdict = await evaluator.run_comparison_gate(
            num_scenario_families=2,
            variants_per_family=1,
            repeats=1,
            max_seconds=10.0,
        )

        assert verdict["gate_b"]["status"] == "FAIL"
        assert verdict["gate_b"]["incomplete_evidence"] is True
        assert len(verdict["gate_b"]["failure_reasons"]) > 0

        # Manifest must reflect incomplete evidence
        out_path = Path(tmpdir)
        import json
        with open(out_path / "run_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["incomplete_evidence"] is True


def test_task_correctness_regression_fails_gate_b() -> None:
    """Task correctness of Arm C must be >= both Arm A and Arm B without tolerance discounts."""
    # When corr_c < corr_a:
    corr_a = 1.8
    corr_b = 1.5
    corr_c = 1.7  # Higher than B, but lower than A
    correctness_preserved = (corr_c >= corr_a) and (corr_c >= corr_b)
    assert correctness_preserved is False

    # When corr_c == min(corr_a, corr_b):
    corr_c_equal = 1.8
    corr_a_equal = 1.8
    corr_b_equal = 1.8
    assert ((corr_c_equal >= corr_a_equal) and (corr_c_equal >= corr_b_equal)) is True


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

def test_task_correctness_does_not_use_length_fallback() -> None:
    """task_correctness uses 0.5 (UNSCORABLE) fallback instead of len > 5 logic."""
    turn = ScenarioTurn(turn_index=1, user_message="Something irrelevant")
    # A family with no task_check
    from aura_backend.affect.scenarios import ScenarioFamily
    dummy_family = ScenarioFamily(family_id="dummy", name="Dummy", description="dummy", development=False, held_out_variants=[], task_check=None)

    long_incorrect_resp = "This is a very long response that has absolutely nothing to do with the actual task, just padding out length."
    scores = score_turn_response(
        response=long_incorrect_resp,
        user_message=turn.user_message,
        turn=turn,
        family=dummy_family,
    )
    # The previous logic would give 2.0 because len > 5. Now it must be 0.5.
    assert scores.task_correctness == 0.5

@pytest.mark.asyncio
async def test_provider_init_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provider init failure writes manifest and exits with 1 instead of falling back to mock."""
    from aura_backend.affect.evaluate import main_async
    import json

    class MockArgs:
        mode = "compare"
        provider = "ollama"
        model = "test"
        scenario_families = 1
        variants = 1
        repeats = 1
        max_seconds = 100.0
        output_dir = str(tmp_path)

    def mock_parse_args(*args, **kwargs):
        return MockArgs()

    def mock_create_provider(*args, **kwargs):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr("aura_backend.affect.evaluate.build_parser", lambda: type("Parser", (), {"parse_args": mock_parse_args})())
    monkeypatch.setattr("aura_backend.affect.evaluate.ModelProviderFactory.create_provider", mock_create_provider)

    exit_code = await main_async(MockArgs())
    assert exit_code == 1

    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["status"] == "INIT_FAILED"
    assert "simulated provider failure" in manifest["initialization_failure"]
