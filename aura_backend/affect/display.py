"""Presentation of controller state, independent of model tone classification."""

from __future__ import annotations

from typing import Any

from aura_backend.affect.models import AffectVector
from aura_backend.affect.policy import compute_channel_readouts


def emotion_readout(state: AffectVector) -> dict[str, str]:
    """Name the strongest authored controller tendency relative to baseline."""
    candidates = {
        "Curious": max(state.novelty, (state.curiosity - 0.60) / 0.40),
        "Excited": min((state.valence - 0.10) / 0.50, (state.arousal - 0.30) / 0.30),
        "Concerned": (state.load - 0.10) / 0.30,
        "Warm": (state.affiliation - 0.65) / 0.35,
        "Content": (state.valence - 0.10) / 0.80,
        "Peaceful": (0.30 - state.arousal) / 0.30,
    }
    name = max(candidates, key=lambda key: candidates[key])
    strength = max(0.0, min(1.0, candidates[name]))
    if strength < 0.10:
        name = "Calm"
    descriptions = {
        "Curious": "Engaged in exploring and understanding.",
        "Excited": "Energized by positive developments.",
        "Concerned": "Attentive care with more unresolved load.",
        "Warm": "Increased warmth and connection.",
        "Content": "Positive and grounded.",
        "Peaceful": "Slowing down and settling.",
        "Calm": "Near the simulation's resting state.",
    }
    return {
        "name": name,
        "intensity": "High" if strength >= 0.65 else "Medium" if strength >= 0.25 else "Low",
        "description": descriptions[name],
    }


def with_simulation_readouts(simulation: dict[str, Any]) -> dict[str, Any]:
    """Attach deterministic display values to a complete published state.

    Wave bands are authored visualization bins for activation, not simulated EEG
    or a neuroscience model. Chemical channels retain the engine's gain formulas.
    Never fabricate a default vector when state is absent or malformed.
    """
    result = dict(simulation)
    result.pop("display", None)
    result.pop("channels", None)
    appraisal = simulation.get("appraisal")
    if isinstance(appraisal, dict) and "analysis_status" in appraisal:
        result["causes"] = appraisal.get("accepted_events", [])
        result["appraisal"] = {
            "status": appraisal["analysis_status"], "reason": appraisal.get("analysis_reason"),
        }
    values = simulation.get("post_state")
    names = (
        "valence",
        "arousal",
        "novelty",
        "affiliation",
        "control",
        "curiosity",
        "load",
    )
    if not isinstance(values, dict) or any(
        isinstance(values.get(name), bool)
        or not isinstance(values.get(name), (int, float))
        for name in names
    ):
        return result
    try:
        state = AffectVector(**{name: values[name] for name in names})
        if state != state.clip():
            return result
    except ValueError:
        return result
    channels = compute_channel_readouts(state)
    band = next(
        name
        for ceiling, name in (
            (0.15, "Delta"),
            (0.30, "Theta"),
            (0.55, "Alpha"),
            (0.80, "Beta"),
            (1.01, "Gamma"),
        )
        if state.arousal < ceiling
    )
    result["channels"] = channels
    result["display"] = {
        "basis": "published_affect_state",
        "brainwave": band,
        "activation": state.arousal,
        "dominant_channel": max(channels, key=lambda name: channels[name]),
        "emotion": emotion_readout(state),
    }
    return result
