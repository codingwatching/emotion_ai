"""Presentation of controller state, independent of model tone classification."""

from __future__ import annotations

from typing import Any

from aura_backend.affect.models import AffectVector
from aura_backend.affect.policy import compute_channel_readouts


def with_simulation_readouts(simulation: dict[str, Any]) -> dict[str, Any]:
    """Attach deterministic display values to a complete published state.

    Wave bands are authored visualization bins for activation, not simulated EEG
    or a neuroscience model. Chemical channels retain the engine's gain formulas.
    Never fabricate a default vector when state is absent or malformed.
    """
    result = dict(simulation)
    result.pop("display", None)
    result.pop("channels", None)
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
    }
    return result
