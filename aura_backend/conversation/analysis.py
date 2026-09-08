"""Provider-neutral analyses with explicit uncertainty and simulated Aura labels."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from aura_backend.conversation.emotion_assessment import (
    EmotionAssessment,
    assess_emotion,
)

from aura_backend.providers.base import ProviderMessage, ProviderRequest, ProviderResult


logger = logging.getLogger(__name__)
ProviderGenerate = Callable[[ProviderRequest], Awaitable[ProviderResult]]


class EmotionalIntensity(str, Enum):
    """Three intensity levels plus an explicit unknown when assessment fails."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


class AsekeComponent(str, Enum):
    """Legacy ASEKE cognitive focus labels."""

    KS = "KS"
    CE = "CE"
    IS = "IS"
    KI = "KI"
    KP = "KP"
    ESA = "ESA"
    SDA = "SDA"
    LEARNING = "Learning"


@dataclass
class EmotionalStateData:
    """Characterized emotional-state record used by routes and persistence."""

    name: str
    formula: str
    components: dict[str, str]
    ntk_layer: str
    brainwave: str
    neurotransmitter: str
    description: str
    intensity: EmotionalIntensity = EmotionalIntensity.MEDIUM
    primary_components: list[str] | None = None
    timestamp: datetime | None = None
    assessment: EmotionAssessment | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CognitiveState:
    """Characterized ASEKE focus record used by routes and persistence."""

    focus: AsekeComponent
    description: str
    context: str
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


# Legacy display analogies are used only for Aura, never as human measurements.
_SIMULATED_TONE_STATES = {
    "Normal": ("Baseline state of calmness", "Alpha", "Serotonin"),
    "Excited": ("Enthusiastic anticipation", "Beta", "Dopamine"),
    "Happy": ("Pleased and content", "Beta", "Endorphin"),
    "Sad": ("Sorrowful or unhappy", "Delta", "Serotonin"),
    "Angry": ("Strong displeasure", "Theta", "Norepinephrine"),
    "Joy": ("Intense happiness", "Gamma", "Oxytocin"),
    "Peace": ("Tranquil and calm", "Theta", "GABA"),
    "Curiosity": ("Strong desire to learn", "Beta", "Dopamine"),
    "Friendliness": ("Kind and warm", "Alpha", "Endorphin"),
    "Love": ("Deep affection", "Alpha", "Oxytocin"),
    "Creativity": ("Inspired and inventive", "Gamma", "Dopamine"),
    "Anxious": ("Worried or nervous", "Beta", "Cortisol"),
    "Tired": ("Exhausted or fatigued", "Delta", "Melatonin"),
}

_ASEKE_COMPONENTS = {
    "KS": "Knowledge Substrate - shared context and history",
    "CE": "Cognitive Energy - focus and mental effort",
    "IS": "Information Structures - ideas and concepts",
    "KI": "Knowledge Integration - connecting new with existing understanding",
    "KP": "Knowledge Propagation - sharing ideas and information",
    "ESA": "Emotional State Algorithms - emotional influence on interaction",
    "SDA": "Sociobiological Drives - social dynamics and trust",
    "Learning": "General learning and information processing",
}


async def _generate_text(prompt: str, generate: ProviderGenerate) -> str:
    """Send one immutable typed request and return normalized result text."""
    result = await generate(
        ProviderRequest(messages=(ProviderMessage(role="user", content=prompt),))
    )
    if not isinstance(result, ProviderResult):
        raise TypeError("analysis provider returned an invalid result")
    return result.content.strip()


def _emotion_state(assessment: EmotionAssessment) -> EmotionalStateData:
    """Adapt a checked proposal to the existing storage DTO without inventing data."""
    name = assessment.emotion or "Unknown"
    brainwave = neurotransmitter = ""
    if assessment.emotion is None:
        descriptions = {
            "abstained": "Not enough clear evidence for an emotion label.",
            "invalid": "Emotion analysis could not be validated.",
            "unavailable": "Emotion analysis is unavailable.",
        }
        description = descriptions.get(assessment.status, "Emotion is unknown.")
    elif assessment.subject == "aura":
        description, brainwave, neurotransmitter = _SIMULATED_TONE_STATES[name]
        description = f"Simulated tone: {description.lower()}. Not measured biology."
    else:
        description = f"Tentative interpretation: {name}. The user may correct it."
    return EmotionalStateData(
        name=name,
        intensity=EmotionalIntensity(assessment.intensity or "Unknown"),
        formula="source_checked_proposal" if assessment.emotion else "unknown",
        components={"basis": assessment.status},
        ntk_layer=f"{brainwave.lower()}-like_NTK" if brainwave else "",
        brainwave=brainwave,
        neurotransmitter=neurotransmitter,
        description=description,
        assessment=assessment,
    )


def emotional_state_payload(
    state: EmotionalStateData | None,
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve public fields while making unknown and legacy states explicit."""
    payload = {
        "name": state.name if state else "Unknown",
        "intensity": state.intensity.value if state else "Unknown",
        "brainwave": state.brainwave if state else "",
        "neurotransmitter": state.neurotransmitter if state else "",
        "description": state.description
        if state
        else "Emotion analysis is unavailable.",
        "assessment": (
            asdict(state.assessment)
            if state and state.assessment
            else {"status": "unverified" if state else "unavailable", "subject": "aura"}
        ),
    }
    if simulation is not None:
        payload["simulation"] = simulation
    return payload


async def detect_user_emotion(
    user_message: str,
    user_id: str,
    *,
    generate: ProviderGenerate,
) -> EmotionalStateData:
    """Return a tentative source-checked user emotion, or an explicit unknown."""
    del user_id
    return _emotion_state(
        await assess_emotion(user_message, subject="user", generate=generate)
    )


async def detect_aura_emotion(
    conversation_snippet: str,
    user_id: str,
    *,
    generate: ProviderGenerate,
) -> EmotionalStateData:
    """Assess Aura's response tone; callers should supply only Aura's visible reply."""
    del user_id
    return _emotion_state(
        await assess_emotion(conversation_snippet, subject="aura", generate=generate)
    )


async def detect_aura_cognitive_focus(
    conversation_snippet: str,
    user_id: str,
    *,
    generate: ProviderGenerate,
) -> CognitiveState | None:
    """Detect Aura's legacy ASEKE focus through the selected provider."""
    del user_id
    components_list = "\n".join(
        f"{code}: {description}" for code, description in _ASEKE_COMPONENTS.items()
    )
    prompt = f"""Analyze this conversation to identify Aura's primary cognitive focus using the ASEKE framework.

ASEKE Components:
{components_list}

Conversation:
{conversation_snippet}

Output only the component code (e.g., "KI", "ESA", "Learning")."""

    try:
        focus_code = await _generate_text(prompt, generate)
        if focus_code in _ASEKE_COMPONENTS:
            return CognitiveState(
                focus=AsekeComponent(focus_code),
                description=_ASEKE_COMPONENTS[focus_code],
                context="Detected from conversation analysis",
            )
        return CognitiveState(
            focus=AsekeComponent.LEARNING,
            description=_ASEKE_COMPONENTS["Learning"],
            context="Default cognitive focus",
        )
    except Exception:
        logger.warning("Aura cognitive-focus analysis unavailable")
        return None
