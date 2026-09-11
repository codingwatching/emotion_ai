"""Causal response policy rendering and functional channel readouts."""

from __future__ import annotations


from aura_backend.affect.models import AffectVector, ResponsePolicy
from aura_backend.affect.regulation import RegulationDecision

WARMTH_LEVELS = ("reserved", "warm", "appreciative")
ENERGY_LEVELS = ("steady", "balanced", "energetic")
EXPLORATION_LEVELS = ("focused", "balanced", "exploratory")


def _clamp_step(candidate: str, prior: str, levels: tuple[str, ...]) -> str:
    """Cap expression changes to adjacent authored levels per committed turn."""
    if candidate not in levels or prior not in levels:
        return candidate
    curr_idx = levels.index(prior)
    cand_idx = levels.index(candidate)
    if cand_idx > curr_idx + 1:
        return levels[curr_idx + 1]
    if cand_idx < curr_idx - 1:
        return levels[curr_idx - 1]
    return candidate


def render_policy(
    state: AffectVector,
    prior_policy: ResponsePolicy | None = None,
    *,
    regulation: RegulationDecision | None = None,
) -> ResponsePolicy:
    """Map affective state vector to authored, causal response policy."""
    # 1. Warmth
    if state.affiliation < 0.45 or state.valence < -0.15:
        target_warmth = "reserved"
    elif state.affiliation > 0.75 and state.valence > 0.12:
        target_warmth = "appreciative"
    else:
        target_warmth = "warm"
    if regulation is not None and target_warmth == "reserved":
        # A leftover negative state is not evidence that this interlocutor
        # deserves distance. Keep the legacy mapping for v1 replay callers.
        target_warmth = "warm"

    # 2. Energy
    if state.arousal < 0.25 or state.load > 0.40:
        target_energy = "steady"
    elif state.arousal > 0.55 and state.load < 0.30:
        target_energy = "energetic"
    else:
        target_energy = "balanced"

    # 3. Exploration
    if state.control > 0.80 or state.curiosity < 0.40 or state.load > 0.35:
        target_exploration = "focused"
    elif state.curiosity > 0.65 and state.control < 0.80 and state.load < 0.30:
        target_exploration = "exploratory"
    else:
        target_exploration = "balanced"

    # Cap to adjacent authored levels if prior policy exists
    if prior_policy is not None:
        warmth = _clamp_step(target_warmth, prior_policy.warmth, WARMTH_LEVELS)
        energy = _clamp_step(target_energy, prior_policy.energy, ENERGY_LEVELS)
        exploration = _clamp_step(target_exploration, prior_policy.exploration, EXPLORATION_LEVELS)
    else:
        warmth = target_warmth
        energy = target_energy
        exploration = target_exploration

    # 4. Setback & recovery
    acknowledge_setback = bool(state.valence < -0.05 and state.load > 0.15)
    recovery_step = bool(acknowledge_setback or (state.valence < -0.02 and state.control >= 0.70))
    evidence_action = regulation.selected_action if regulation is not None else "proceed"
    if regulation is not None:
        acknowledge_setback = any(
            interpretation.validation_status == "verified"
            and interpretation.current_consequence == "active_error"
            for interpretation in regulation.accepted_interpretations
        )
        recovery_step = acknowledge_setback

    # 5. Reflection
    reflection = "immediate" if (state.load < 0.25 and state.curiosity > 0.65) else "defer"

    # 6. Render compact, server-authored policy instruction
    prompt_lines = [
        "### Simulated Behavioral Posture (affect-v1)",
        f"- Warmth: {warmth} (maintain warmth, candor, and playfulness without gratuitous emotional narration)",
        f"- Energy & Pace: {energy}",
        f"- Setback Disposition: {'Acknowledge the setback calmly and take a deliberate recovery step' if acknowledge_setback else 'Proceed normally'}",
        f"- Initiative & Exploration: {exploration} (keep factual and task requirements strictly prioritized)",
        f"- Reflection: {reflection}",
    ]
    if evidence_action == "verify":
        prompt_lines.append("- Evidence handling: Check the claim against available task evidence before accepting it; if no check is available, keep it explicitly unverified.")
    prompt_block = "\n".join(prompt_lines)

    return ResponsePolicy(
        warmth=warmth,
        energy=energy,
        acknowledge_setback=acknowledge_setback,
        recovery_step=recovery_step,
        exploration=exploration,
        reflection=reflection,
        prompt_block=prompt_block,
        input_state=state,
        evidence_action=evidence_action,
    )


def compute_channel_readouts(state: AffectVector) -> dict[str, float]:
    """Calculate functional controller gains for chemical channel analogies.

    These are readouts of the authoritative underlying state, not independent mystery variables.
    """
    # Dopamine-like: approach / exploration readiness
    dopamine_like = max(0.0, min(1.0, 0.5 * (state.valence + 1.0) * 0.5 + 0.5 * state.curiosity))

    # Norepinephrine-like: novelty + arousal vigilance
    norepinephrine_like = max(0.0, min(1.0, 0.5 * state.novelty + 0.5 * state.arousal))

    # Acetylcholine-like: uncertainty / evidence inspection priority
    acetylcholine_like = max(0.0, min(1.0, 0.6 * state.novelty + 0.4 * (1.0 - state.control)))

    # Serotonin-like: slow stability / baseline groundedness
    serotonin_like = max(0.0, min(1.0, 0.6 * state.affiliation + 0.4 * (1.0 - state.load)))

    # GABA-like: impulse restraint and deliberate control
    gaba_like = max(0.0, min(1.0, state.control))

    # Cortisol-like: accumulated unresolved load
    cortisol_like = max(0.0, min(1.0, state.load))

    return {
        "dopamine_like": round(dopamine_like, 4),
        "norepinephrine_like": round(norepinephrine_like, 4),
        "acetylcholine_like": round(acetylcholine_like, 4),
        "serotonin_like": round(serotonin_like, 4),
        "gaba_like": round(gaba_like, 4),
        "cortisol_like": round(cortisol_like, 4),
    }
