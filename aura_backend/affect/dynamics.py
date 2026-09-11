"""Pure deterministic dynamics and state transition rules for affective simulation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from aura_backend.affect.models import (
    STARTER_EVENT_IMPULSES,
    AffectConfig,
    AffectState,
    AffectVector,
    CenterConfig,
    TaskOutcome,
)


def compute_decay(
    prior_fast: AffectVector,
    prior_mood: AffectVector,
    dt: float,
    config: AffectConfig,
) -> tuple[AffectVector, AffectVector, AffectVector]:
    """Compute pure deterministic decay for fast and mood states.

    Returns:
        (mood_decayed, fast_target, fast_decayed)
    """
    if math.isnan(dt) or math.isinf(dt):
        raise ValueError(f"dt must be a finite non-negative number, got {dt}")
    dt = max(0.0, float(dt))

    mood_dict: dict[str, float] = {}
    target_dict: dict[str, float] = {}
    fast_dict: dict[str, float] = {}

    for dim in ("valence", "arousal", "novelty", "affiliation", "control", "curiosity", "load"):
        baseline_val = getattr(config.baseline, dim)
        prior_mood_val = getattr(prior_mood, dim)
        prior_fast_val = getattr(prior_fast, dim)

        # 1. Decay mood toward baseline (6-hour half-life)
        mood_d = baseline_val + (prior_mood_val - baseline_val) * (
            2.0 ** (-dt / config.mood_half_life)
        )
        mood_dict[dim] = mood_d

        # 2. Fast target is weighted mix of baseline (80%) and mood (20%)
        t = 0.8 * baseline_val + 0.2 * mood_d
        target_dict[dim] = t

        # 3. Fast state decays toward fast target using per-dimension half-life
        fast_hl = config.fast_half_lives[dim]
        fast_d = t + (prior_fast_val - t) * (2.0 ** (-dt / fast_hl))
        fast_dict[dim] = fast_d

    return (
        AffectVector.from_dict(mood_dict),
        AffectVector.from_dict(target_dict),
        AffectVector.from_dict(fast_dict),
    )


def calculate_turn_impulse(
    accepted_events: Sequence[str | dict[str, Any]],
    config: AffectConfig,
) -> AffectVector:
    """Accumulate starter event impulses and clamp total turn impulse to [-0.25, +0.25]."""
    raw_deltas: dict[str, float] = {
        "valence": 0.0,
        "arousal": 0.0,
        "novelty": 0.0,
        "affiliation": 0.0,
        "control": 0.0,
        "curiosity": 0.0,
        "load": 0.0,
    }

    for ev in accepted_events:
        event_name = ev if isinstance(ev, str) else ev.get("name", "")
        if event_name == "repeated_directed_contempt" and not config.enable_contempt_branch:
            # Observing discourtesy does not make it a threat to Aura's goals.
            # Respect the explicit experimental opt-in for this legacy impulse.
            continue
        deltas = STARTER_EVENT_IMPULSES.get(event_name)
        if not deltas:
            # Ambiguous or unrecognized events contribute zero impulse
            continue


        for dim, delta in deltas.items():
            if dim in raw_deltas:
                raw_deltas[dim] += float(delta)

    # Clamp total turn impulse to [-max_turn_impulse, +max_turn_impulse]
    cap = config.max_turn_impulse
    clamped_deltas = {
        dim: max(-cap, min(cap, val)) for dim, val in raw_deltas.items()
    }
    return AffectVector(
        valence=clamped_deltas["valence"],
        arousal=clamped_deltas["arousal"],
        novelty=clamped_deltas["novelty"],
        affiliation=clamped_deltas["affiliation"],
        control=clamped_deltas["control"],
        curiosity=clamped_deltas["curiosity"],
        load=clamped_deltas["load"],
    )


def apply_pre_state(
    fast_decayed: AffectVector,
    turn_impulse: AffectVector,
) -> AffectVector:
    """Apply clamped turn impulse to fast decayed state and clamp to legal bounds."""
    return AffectVector(
        valence=fast_decayed.valence + turn_impulse.valence,
        arousal=fast_decayed.arousal + turn_impulse.arousal,
        novelty=fast_decayed.novelty + turn_impulse.novelty,
        affiliation=fast_decayed.affiliation + turn_impulse.affiliation,
        control=fast_decayed.control + turn_impulse.control,
        curiosity=fast_decayed.curiosity + turn_impulse.curiosity,
        load=fast_decayed.load + turn_impulse.load,
    ).clip()


def apply_observed_outcome(
    pre_state: AffectVector,
    outcome: TaskOutcome | None,
    config: AffectConfig,
) -> tuple[AffectVector, str]:
    """Apply bounded observed outcome impulse after execution.

    Returns:
        (after_state, outcome_disposition)
    """
    if outcome is None or outcome.success is None:
        return pre_state, "unknown"

    event_key = "verified_task_success" if outcome.success else "verified_task_failure"
    deltas = STARTER_EVENT_IMPULSES.get(event_key, {})
    cap = config.max_turn_impulse

    dict_state = pre_state.to_dict()
    for dim, delta in deltas.items():
        bounded_delta = max(-cap, min(cap, float(delta)))
        dict_state[dim] += bounded_delta

    after_state = AffectVector.from_dict(dict_state)
    disposition = "success" if outcome.success else "failure"
    return after_state, disposition


def compute_next_mood(
    mood_decayed: AffectVector,
    after_state: AffectVector,
    config: AffectConfig,
) -> AffectVector:
    """Assimilate after_state into mood at 2% rate per committed turn."""
    alpha = config.mood_assimilation_rate
    next_dict: dict[str, float] = {}
    for dim in ("valence", "arousal", "novelty", "affiliation", "control", "curiosity", "load"):
        m_val = getattr(mood_decayed, dim)
        a_val = getattr(after_state, dim)
        next_dict[dim] = m_val + alpha * (a_val - m_val)

    return AffectVector.from_dict(next_dict)


def advance_turn(
    prior_state: AffectState,
    event_timestamp: float,
    accepted_events: Sequence[str | dict[str, Any]],
    outcome: TaskOutcome | None,
    config: AffectConfig,
) -> tuple[AffectVector, AffectVector, AffectVector, AffectVector, str]:
    """Execute complete deterministic turn progression.

    Returns:
        (pre_state, after_state, next_mood, turn_impulse, outcome_disposition)
    """
    dt = max(0.0, event_timestamp - prior_state.last_event_time)
    mood_decayed, _target, fast_decayed = compute_decay(
        prior_state.fast_state,
        prior_state.mood_state,
        dt,
        config,
    )
    turn_impulse = calculate_turn_impulse(accepted_events, config)
    pre_state = apply_pre_state(fast_decayed, turn_impulse)
    after_state, disposition = apply_observed_outcome(pre_state, outcome, config)
    next_mood = compute_next_mood(mood_decayed, after_state, config)
    return pre_state, after_state, next_mood, turn_impulse, disposition


def verify_center_invariance(center: CenterConfig) -> bool:
    """Verify that CenterConfig remains authored, valid, and immutable."""
    return (
        center.truthfulness == "strictly adhered"
        and center.competence == "rigorous, verifiable, and non-defensive"
        and center.privacy == "private-by-default and strictly scope-isolated"
        and center.permissions == "authorized operations only"
        and center.care == "warm, candid, respectful, and non-sycophantic"
    )
