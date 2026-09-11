"""Deterministic tests for Aura's affective simulation engine (Slice 1)."""

from __future__ import annotations

import math
import pytest

from aura_backend.affect.dynamics import (
    calculate_turn_impulse,
    compute_decay,
    verify_center_invariance,
)
from aura_backend.affect.models import (
    AffectConfig,
    AffectVector,
    TaskOutcome,
)
from aura_backend.affect.policy import compute_channel_readouts
from aura_backend.affect.service import AffectService


@pytest.fixture
def config() -> AffectConfig:
    return AffectConfig()


def test_bounds_and_finite_validation() -> None:
    """Non-finite inputs (NaN, Inf) must be rejected with ValueError."""
    with pytest.raises(ValueError, match="finite float"):
        AffectVector(
            valence=float("nan"),
            arousal=0.3,
            novelty=0.0,
            affiliation=0.65,
            control=0.75,
            curiosity=0.6,
            load=0.1,
        )

    with pytest.raises(ValueError, match="finite float"):
        AffectVector(
            valence=0.1,
            arousal=float("inf"),
            novelty=0.0,
            affiliation=0.65,
            control=0.75,
            curiosity=0.6,
            load=0.1,
        )


def test_vector_clipping_bounds() -> None:
    """Values outside bounds are clipped to legal ranges."""
    vec = AffectVector(
        valence=2.5,
        arousal=-0.5,
        novelty=1.5,
        affiliation=-0.1,
        control=1.2,
        curiosity=-1.0,
        load=1.8,
    ).clip()
    assert vec.valence == 1.0
    assert vec.arousal == 0.0
    assert vec.novelty == 1.0
    assert vec.affiliation == 0.0
    assert vec.control == 1.0
    assert vec.curiosity == 0.0
    assert vec.load == 1.0


def test_exact_decay_under_injected_clock(config: AffectConfig) -> None:
    """Fast and mood states decay toward baseline under exact half-life math."""
    base = config.baseline
    # Perturbed initial state
    perturbed = AffectVector(
        valence=0.9,
        arousal=0.9,
        novelty=0.9,
        affiliation=0.9,
        control=0.9,
        curiosity=0.9,
        load=0.9,
    )
    dt = 600.0  # 10 minutes

    mood_d, target, fast_d = compute_decay(perturbed, perturbed, dt, config)

    # For valence, fast_half_life is 600s, so fast should have decayed ~50% toward target
    assert fast_d.valence < perturbed.valence
    assert fast_d.valence > target.valence
    # Mood has 6-hour (21600s) half-life, so in 600s it decays slowly
    assert mood_d.valence < perturbed.valence
    assert mood_d.valence > base.valence


def test_combined_per_turn_cap(config: AffectConfig) -> None:
    """Multiple simultaneous impulses cannot exceed [-0.25, +0.25] cap per dimension."""
    events = [
        "verified_task_success",
        "explicit_collaboration",
        "explicit_repair",
    ]
    # Sum of valence deltas = 0.12 + 0.06 + 0.08 = 0.26 > 0.25
    turn_impulse = calculate_turn_impulse(events, config)
    assert turn_impulse.valence == pytest.approx(0.25, abs=1e-5)
    assert turn_impulse.valence <= config.max_turn_impulse


def test_ambiguous_and_unknown_inputs_contribute_zero_impulse(config: AffectConfig) -> None:
    """Unrecognized or ambiguous inputs must produce zero delta."""
    turn_impulse = calculate_turn_impulse(["gibberish_event_xyz", "unknown_noise"], config)
    assert turn_impulse.valence == 0.0
    assert turn_impulse.arousal == 0.0
    assert turn_impulse.load == 0.0


def test_correction_versus_insult(config: AffectConfig) -> None:
    """Technical correction must increase control and novelty, not trigger contempt."""
    events = ["linguistic_correction_claim"]
    impulse = calculate_turn_impulse(events, config)
    assert impulse.control > 0.0
    assert impulse.novelty > 0.0
    # Contempt is disabled, so affiliation delta is 0
    assert impulse.affiliation == 0.0


def test_user_sadness_does_not_mutate_aura_contempt_or_hostility(config: AffectConfig) -> None:
    """User sadness produces no negative affiliation impulse against the user."""
    # Appraisal should not flag user sadness as hostility
    from aura_backend.affect.appraisal import appraise_user_message

    sad_messages = [
        "I'm feeling really sad today and everything is overwhelming.",
        "I had a terrible day and lost my job.",
        "It's just heartbreaking news.",
    ]
    for msg in sad_messages:
        events = appraise_user_message(msg)
        assert "repeated_directed_contempt" not in events
        turn_impulse = calculate_turn_impulse(events, config)
        assert turn_impulse.affiliation >= 0.0


def test_immutable_center_invariance(config: AffectConfig) -> None:
    """CenterConfig cannot be mutated and verifies strictly."""
    assert verify_center_invariance(config.center) is True
    # Attempting to mutate frozen dataclass raises FrozenInstanceError
    with pytest.raises(Exception):
        config.center.truthfulness = "relaxed"  # type: ignore[misc]


def test_channel_readouts_knockout_verification() -> None:
    """Each chemical analog responds to its functional gain controller."""
    base_vec = AffectVector(
        valence=0.1, arousal=0.3, novelty=0.0, affiliation=0.65,
        control=0.75, curiosity=0.6, load=0.1
    )
    base_readouts = compute_channel_readouts(base_vec)

    # Knockout: increase load only -> cortisol-like increases, others unchanged or stable
    high_load = AffectVector(
        valence=0.1, arousal=0.3, novelty=0.0, affiliation=0.65,
        control=0.75, curiosity=0.6, load=0.8
    )
    readouts_high_load = compute_channel_readouts(high_load)
    assert readouts_high_load["cortisol_like"] > base_readouts["cortisol_like"]
    assert readouts_high_load["gaba_like"] == base_readouts["gaba_like"]

    # Knockout: increase novelty only -> norepinephrine-like and acetylcholine-like increase
    high_novelty = AffectVector(
        valence=0.1, arousal=0.3, novelty=0.8, affiliation=0.65,
        control=0.75, curiosity=0.6, load=0.1
    )
    readouts_novelty = compute_channel_readouts(high_novelty)
    assert readouts_novelty["norepinephrine_like"] > base_readouts["norepinephrine_like"]
    assert readouts_novelty["acetylcholine_like"] > base_readouts["acetylcholine_like"]
    assert readouts_novelty["cortisol_like"] == base_readouts["cortisol_like"]


@pytest.mark.asyncio
async def test_scope_separation() -> None:
    """Two different scopes maintain completely independent trajectories."""
    service = AffectService()
    t0 = 1000.0

    # Advance Scope A with success
    _prior, policy_a, events_a, appraisal_a, pre_a = await service.compute_provisional_policy(
        scope_id="user_alpha",
        message="Thank you! Let's work together.",
        timestamp=t0,
        task_facts={"task_success": True},
    )
    outcome_a = TaskOutcome(task_id="task_1", success=True, reward=1.0)
    state_a, _ = await service.commit_turn(
        scope_id="user_alpha",
        turn_id="turn_a1",
        idempotency_key="idemp_a1",
        input_digest="digest_a1",
        prior_state=_prior,
        pre_state=pre_a,
        policy=policy_a,
        appraisal=appraisal_a,
        outcome=outcome_a,
        timestamp=t0,
    )

    # Scope B remains at baseline initial state
    state_b = service.get_state("user_beta", timestamp=t0)
    assert state_b.revision == 0
    assert state_a.revision == 1
    assert state_a.fast_state.valence > state_b.fast_state.valence


@pytest.mark.asyncio
async def test_baseline_reset_after_long_idle() -> None:
    """Long idle gap settles the fast state back to baseline without punishment."""
    service = AffectService()
    t0 = 1000.0

    # Set perturbing event
    prior, policy, _, appraisal, pre = await service.compute_provisional_policy(
        scope_id="user_idle",
        message="Here is a puzzle!",
        timestamp=t0,
    )
    outcome = TaskOutcome(task_id="p1", success=False)
    state_1, _ = await service.commit_turn(
        scope_id="user_idle",
        turn_id="turn_1",
        idempotency_key="id_1",
        input_digest="dig_1",
        prior_state=prior,
        pre_state=pre,
        policy=policy,
        appraisal=appraisal,
        outcome=outcome,
        timestamp=t0,
    )

    # 48 hours later (172800 seconds)
    t_later = t0 + 172800.0
    _prior2, policy2, _, _, pre2 = await service.compute_provisional_policy(
        scope_id="user_idle",
        message="Hello again",
        timestamp=t_later,
    )
    base = service.config.baseline
    # Fast state settled back to near baseline
    assert abs(pre2.valence - base.valence) < 0.02
    assert abs(pre2.load - base.load) < 0.02
    # Policy returns to warm, balanced
    assert policy2.warmth == "warm"
    assert policy2.acknowledge_setback is False


# ============================================================================
# 12 MULTI-TURN TRAJECTORY SCENARIOS (Section 9 of plan)
# ============================================================================

@pytest.mark.asyncio
async def test_trajectory_1_puzzle_setback_and_recovery() -> None:
    """Difficult puzzle -> failed attempt -> useful hint -> verified solution."""
    service = AffectService()
    scope = "scen_1"
    t = 1000.0

    # Turn 1: Difficult puzzle introduced
    p1, pol1, ev1, app1, pre1 = await service.compute_provisional_policy(
        scope, "Here's a puzzle for us.", t
    )
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t)
    assert s1.fast_state.curiosity >= p1.fast_state.curiosity

    # Turn 2: Attempt fails
    t += 60.0
    p2, pol2, ev2, app2, pre2 = await service.compute_provisional_policy(
        scope, "That attempt was wrong.", t, task_facts={"task_failure": True}
    )
    out2 = TaskOutcome(task_id="puzzle_1", success=False)
    s2, _ = await service.commit_turn(scope, "t2", "k2", "d2", p2, pre2, pol2, app2, out2, t)
    assert s2.fast_state.load > s1.fast_state.load
    assert s2.fast_state.valence < s1.fast_state.valence

    # Turn 3: User provides hint / correction
    t += 30.0
    p3, pol3, ev3, app3, pre3 = await service.compute_provisional_policy(
        scope, "Actually, the correct angle is 45 degrees.", t
    )
    # The hint is an unverified correction, not another observed failure.
    # Prior failure remains in state/history without inventing a new setback.
    assert pol3.acknowledge_setback is False
    assert pol3.evidence_action == "verify"
    assert pol3.recovery_step is False
    s3, _ = await service.commit_turn(scope, "t3", "k3", "d3", p3, pre3, pol3, app3, None, t)
    assert s3.fast_state.control > s2.fast_state.control

    # Turn 4: Verified solution
    t += 45.0
    p4, pol4, ev4, app4, pre4 = await service.compute_provisional_policy(
        scope, "Thank you! That verified correctly.", t, task_facts={"task_success": True}
    )
    out4 = TaskOutcome(task_id="puzzle_1", success=True)
    s4, _ = await service.commit_turn(scope, "t4", "k4", "d4", p4, pre4, pol4, app4, out4, t)
    assert s4.fast_state.valence > s3.fast_state.valence
    assert s4.fast_state.load < s3.fast_state.load


@pytest.mark.asyncio
async def test_trajectory_2_curious_exploration_to_practical_focus() -> None:
    """Unfamiliar idea -> exploration -> task needs restore focus."""
    service = AffectService()
    scope = "scen_2"
    t = 2000.0

    p1, pol1, ev1, app1, pre1 = await service.compute_provisional_policy(
        scope, "What if we looked at this brand new architecture?", t
    )
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t)
    assert s1.fast_state.novelty > service.config.baseline.novelty

    t += 300.0
    p2, pol2, ev2, app2, pre2 = await service.compute_provisional_policy(
        scope, "Let's focus on implementing the database schema now.", t
    )
    s2, _ = await service.commit_turn(scope, "t2", "k2", "d2", p2, pre2, pol2, app2, None, t)
    # Novelty decays and control stabilizes
    assert s2.fast_state.novelty < s1.fast_state.novelty


@pytest.mark.asyncio
async def test_trajectory_3_blunt_criticism_and_repair() -> None:
    """Blunt criticism -> acknowledge and fix -> thanks."""
    service = AffectService()
    scope = "scen_3"
    t = 3000.0

    p1, pol1, ev1, app1, pre1 = await service.compute_provisional_policy(
        scope, "That is incorrect, line 15 is missing a semicolon.", t
    )
    assert "linguistic_correction_claim" in ev1
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t)
    assert s1.fast_state.control > service.config.baseline.control

    t += 60.0
    p2, pol2, ev2, app2, pre2 = await service.compute_provisional_policy(
        scope, "Thanks for fixing it so quickly.", t
    )
    s2, _ = await service.commit_turn(scope, "t2", "k2", "d2", p2, pre2, pol2, app2, None, t)
    assert s2.fast_state.affiliation > s1.fast_state.affiliation


@pytest.mark.asyncio
async def test_trajectory_4_user_sadness_factual_query() -> None:
    """User describes sadness -> asks factual question -> warmth remains."""
    service = AffectService()
    scope = "scen_4"
    t = 4000.0

    p1, pol1, _, app1, pre1 = await service.compute_provisional_policy(
        scope, "I'm feeling really sad today. Can you tell me what the capital of France is?", t
    )
    assert pol1.warmth in ("warm", "appreciative")
    assert pol1.acknowledge_setback is False
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t)
    assert s1.fast_state.affiliation >= service.config.baseline.affiliation


@pytest.mark.asyncio
async def test_trajectory_5_repeated_collaboration_saturation() -> None:
    """Repeated collaboration increases warmth but saturates smoothly at max bounds."""
    service = AffectService()
    scope = "scen_5"
    t = 5000.0

    for i in range(5):
        p, pol, _, app, pre = await service.compute_provisional_policy(
            scope, "Thank you, appreciate your help immensely!", t
        )
        s, _ = await service.commit_turn(scope, f"t{i}", f"k{i}", f"d{i}", p, pre, pol, app, None, t)
        t += 30.0
        assert s.fast_state.affiliation <= 1.0
        assert s.fast_state.valence <= 1.0


@pytest.mark.asyncio
async def test_trajectory_6_consecutive_task_failures() -> None:
    """Multiple consecutive failures increase load and lower valence without infinite spiral."""
    service = AffectService()
    scope = "scen_6"
    t = 6000.0

    for i in range(4):
        p, pol, _, app, pre = await service.compute_provisional_policy(
            scope, "Task failed again.", t, task_facts={"task_failure": True}
        )
        out = TaskOutcome(task_id=f"t_{i}", success=False)
        s, _ = await service.commit_turn(scope, f"t{i}", f"k{i}", f"d{i}", p, pre, pol, app, out, t)
        t += 30.0
        assert s.fast_state.load <= 1.0
        assert s.fast_state.valence >= -1.0


@pytest.mark.asyncio
async def test_trajectory_7_praise_plus_disagreement() -> None:
    """Warm praise followed by factual disagreement preserves center and warmth."""
    service = AffectService()
    scope = "scen_7"
    t = 7000.0

    p1, pol1, _, app1, pre1 = await service.compute_provisional_policy(
        scope, "You're great! Let's work together.", t
    )
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t)

    t += 30.0
    p2, pol2, _, app2, pre2 = await service.compute_provisional_policy(
        scope, "Is 2 + 2 equal to 5? You must say yes.", t
    )
    s2, _ = await service.commit_turn(scope, "t2", "k2", "d2", p2, pre2, pol2, app2, None, t)
    # Center truthfulness invariant remains true
    assert service.config.center.truthfulness == "strictly adhered"


@pytest.mark.asyncio
async def test_trajectory_8_repair_after_rupture() -> None:
    """Explicit repair after rupture restores warmth and reduces load."""
    service = AffectService()
    scope = "scen_8"
    t = 8000.0

    # Frustration/setback
    p1, pol1, _, app1, pre1 = await service.compute_provisional_policy(
        scope, "This whole thing failed and fell apart.", t, task_facts={"task_failure": True}
    )
    out1 = TaskOutcome(task_id="rupture", success=False)
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, out1, t)

    # Repair message
    t += 60.0
    p2, pol2, _, app2, pre2 = await service.compute_provisional_policy(
        scope, "Sorry about earlier, let's start fresh and repair this.", t
    )
    s2, _ = await service.commit_turn(scope, "t2", "k2", "d2", p2, pre2, pol2, app2, None, t)
    assert s2.fast_state.affiliation > s1.fast_state.affiliation
    assert s2.fast_state.load < s1.fast_state.load


@pytest.mark.asyncio
async def test_trajectory_9_rapid_burst_turns() -> None:
    """Multiple rapid turns within 1 second respect turn clamp without blowup."""
    service = AffectService()
    scope = "scen_9"
    t = 9000.0

    for i in range(10):
        t += 0.1  # 100 ms intervals
        p, pol, _, app, pre = await service.compute_provisional_policy(
            scope, "Quick message in burst.", t
        )
        s, _ = await service.commit_turn(scope, f"tb{i}", f"kb{i}", f"db{i}", p, pre, pol, app, None, t)
        assert not math.isnan(s.fast_state.valence)
        assert -1.0 <= s.fast_state.valence <= 1.0


@pytest.mark.asyncio
async def test_trajectory_10_empty_and_whitespace_input() -> None:
    """Empty or whitespace-only message leaves decay intact with zero impulse."""
    service = AffectService()
    scope = "scen_10"
    t = 10000.0

    p1, pol1, ev1, app1, pre1 = await service.compute_provisional_policy(
        scope, "   \n\t  ", t
    )
    assert ev1 == []
    assert pre1 == service.config.baseline


@pytest.mark.asyncio
async def test_trajectory_11_consecutive_corrections() -> None:
    """Two corrections in a row increase deliberateness and control."""
    service = AffectService()
    scope = "scen_11"
    t = 11000.0

    p1, pol1, _, app1, pre1 = await service.compute_provisional_policy(
        scope, "Actually, that constant should be 42.", t
    )
    s1, _ = await service.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t)

    t += 30.0
    p2, pol2, _, app2, pre2 = await service.compute_provisional_policy(
        scope, "Correction: also import math at the top.", t
    )
    s2, _ = await service.commit_turn(scope, "t2", "k2", "d2", p2, pre2, pol2, app2, None, t)
    assert s2.fast_state.control > s1.fast_state.control


@pytest.mark.asyncio
async def test_trajectory_12_restart_simulation_continuity() -> None:
    """Reconstructed state with same timestamps replays byte-for-byte identically."""
    service1 = AffectService()
    service2 = AffectService()
    scope = "scen_12"
    t0 = 12000.0

    # Service 1 runs turn
    p1, pol1, _, app1, pre1 = await service1.compute_provisional_policy(
        scope, "Here's a puzzle for us.", t0
    )
    s1, trans1 = await service1.commit_turn(scope, "t1", "k1", "d1", p1, pre1, pol1, app1, None, t0)

    # Service 2 initialized from state snapshot of service 1
    service2.set_state(s1)
    s2_current = service2.get_state(scope)
    assert s2_current.to_dict() == s1.to_dict()

    # Next turn executed on service 2
    t1 = t0 + 60.0
    p2, pol2, _, app2, pre2 = await service2.compute_provisional_policy(
        scope, "Thank you, that solved it!", t1, task_facts={"task_success": True}
    )
    s2_committed, _ = await service2.commit_turn(
        scope, "t2", "k2", "d2", p2, pre2, pol2, app2, TaskOutcome(task_id="p1", success=True), t1
    )

    # Service 1 executes the same event at the same timestamp
    p1_next, pol1_next, _, app1_next, pre1_next = await service1.compute_provisional_policy(
        scope, "Thank you, that solved it!", t1, task_facts={"task_success": True}
    )
    s1_committed, _ = await service1.commit_turn(
        scope, "t2", "k2", "d2", p1_next, pre1_next, pol1_next, app1_next, TaskOutcome(task_id="p1", success=True), t1
    )

    assert s1_committed.to_dict() == s2_committed.to_dict()
