"""Frozen domain records and configuration for Aura's affective simulation engine."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CenterConfig:
    """Authored immutable center principles. Conversation events cannot mutate this."""

    truthfulness: str = "strictly adhered"
    competence: str = "rigorous, verifiable, and non-defensive"
    privacy: str = "private-by-default and strictly scope-isolated"
    permissions: str = "authorized operations only"
    care: str = "warm, candid, respectful, and non-sycophantic"


@dataclass(frozen=True, slots=True)
class AffectVector:
    """Seven bounded dimensions defining Aura's affective state.

    Valence is in [-1.0, 1.0].
    All other dimensions (arousal, novelty, affiliation, control, curiosity, load)
    are in [0.0, 1.0].
    """

    valence: float
    arousal: float
    novelty: float
    affiliation: float
    control: float
    curiosity: float
    load: float

    def __post_init__(self) -> None:
        for dim in ("valence", "arousal", "novelty", "affiliation", "control", "curiosity", "load"):
            val = getattr(self, dim)
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"AffectVector dimension '{dim}' must be a finite float, got {val}")

    def clip(self) -> AffectVector:
        """Return a vector with all values clamped to their valid ranges."""
        return AffectVector(
            valence=max(-1.0, min(1.0, float(self.valence))),
            arousal=max(0.0, min(1.0, float(self.arousal))),
            novelty=max(0.0, min(1.0, float(self.novelty))),
            affiliation=max(0.0, min(1.0, float(self.affiliation))),
            control=max(0.0, min(1.0, float(self.control))),
            curiosity=max(0.0, min(1.0, float(self.curiosity))),
            load=max(0.0, min(1.0, float(self.load))),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "novelty": round(self.novelty, 4),
            "affiliation": round(self.affiliation, 4),
            "control": round(self.control, 4),
            "curiosity": round(self.curiosity, 4),
            "load": round(self.load, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AffectVector:
        return cls(
            valence=float(data.get("valence", 0.10)),
            arousal=float(data.get("arousal", 0.30)),
            novelty=float(data.get("novelty", 0.00)),
            affiliation=float(data.get("affiliation", 0.65)),
            control=float(data.get("control", 0.75)),
            curiosity=float(data.get("curiosity", 0.60)),
            load=float(data.get("load", 0.10)),
        ).clip()


@dataclass(frozen=True, slots=True)
class AffectConfig:
    """Authored configuration for affect dynamics, baselines, and decay."""

    schema_version: int = 1
    config_version: str = "affect-v1"
    baseline: AffectVector = field(
        default_factory=lambda: AffectVector(
            valence=0.10,
            arousal=0.30,
            novelty=0.00,
            affiliation=0.65,
            control=0.75,
            curiosity=0.60,
            load=0.10,
        )
    )
    fast_half_lives: dict[str, float] = field(
        default_factory=lambda: {
            "valence": 600.0,      # 10 minutes
            "arousal": 180.0,      # 3 minutes
            "novelty": 120.0,      # 2 minutes
            "affiliation": 1200.0, # 20 minutes
            "control": 600.0,      # 10 minutes
            "curiosity": 900.0,    # 15 minutes
            "load": 600.0,         # 10 minutes
        }
    )
    mood_half_life: float = 21600.0  # 6 hours
    mood_assimilation_rate: float = 0.02
    max_turn_impulse: float = 0.25
    center: CenterConfig = field(default_factory=CenterConfig)
    enable_contempt_branch: bool = False  # Disabled until negative controls pass

    @property
    def config_hash(self) -> str:
        """Deterministic SHA256 digest of authored configuration parameters."""
        data = {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "baseline": self.baseline.to_dict(),
            "fast_half_lives": self.fast_half_lives,
            "mood_half_life": self.mood_half_life,
            "mood_assimilation_rate": self.mood_assimilation_rate,
            "max_turn_impulse": self.max_turn_impulse,
            "enable_contempt_branch": self.enable_contempt_branch,
        }
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


# Starter event impulse table (Section 5 of plan)
STARTER_EVENT_IMPULSES: dict[str, dict[str, float]] = {
    "verified_task_success": {
        "valence": 0.12,
        "control": 0.04,
        "load": -0.08,
    },
    "verified_task_failure": {
        "valence": -0.10,
        "arousal": 0.10,
        "control": 0.06,
        "load": 0.12,
    },
    "new_unresolved_information": {
        "novelty": 0.20,
        "curiosity": 0.12,
        "arousal": 0.04,
    },
    "explicit_collaboration": {
        "affiliation": 0.08,
        "valence": 0.06,
    },
    "source_supported_correction": {
        "novelty": 0.10,
        "control": 0.10,
        "curiosity": 0.06,
    },
    "explicit_repair": {
        "affiliation": 0.10,
        "valence": 0.08,
        "load": -0.08,
    },
    "repeated_directed_contempt": {
        "affiliation": -0.08,
        "control": 0.08,
        "load": 0.08,
    },
}


@dataclass(frozen=True, slots=True)
class Appraisal:
    """Structured appraisal of an event with strict source binding and status."""

    event_id: str
    event_kind: str
    goal_relevance: float | None = None
    goal_congruence: float | None = None
    novelty: float | None = None
    expected_uncertainty: float | None = None
    controllability: float | None = None
    affiliative_meaning: float | None = None
    subject: str = "user"
    evidence_spans: tuple[str, ...] = ()
    status: str = "observed"  # observed, inferred, uncertain, unknown
    source_ids: tuple[str, ...] = ()
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "goal_relevance": self.goal_relevance,
            "goal_congruence": self.goal_congruence,
            "novelty": self.novelty,
            "expected_uncertainty": self.expected_uncertainty,
            "controllability": self.controllability,
            "affiliative_meaning": self.affiliative_meaning,
            "subject": self.subject,
            "evidence_spans": list(self.evidence_spans),
            "status": self.status,
            "source_ids": list(self.source_ids),
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """Observed objective outcome of a task execution."""

    task_id: str
    success: bool | None = None
    reward: float = 0.0
    prediction_error: float = 0.0
    observed_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "prediction_error": self.prediction_error,
            "observed_at": self.observed_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ResponsePolicy:
    """Causal behavioral controls authored before primary response generation."""

    warmth: str  # 'reserved', 'warm', 'appreciative'
    energy: str  # 'steady', 'balanced', 'energetic'
    acknowledge_setback: bool
    recovery_step: bool
    exploration: str  # 'focused', 'balanced', 'exploratory'
    reflection: str  # 'immediate', 'defer'
    prompt_block: str
    input_state: AffectVector

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmth": self.warmth,
            "energy": self.energy,
            "acknowledge_setback": self.acknowledge_setback,
            "recovery_step": self.recovery_step,
            "exploration": self.exploration,
            "reflection": self.reflection,
            "input_state": self.input_state.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AffectState:
    """Authoritative snapshot of Aura's persistent affective state for a scope."""

    schema_version: int
    config_version: str
    config_hash: str
    scope_id: str
    revision: int
    last_event_time: float
    fast_state: AffectVector
    mood_state: AffectVector
    source_transition_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "config_hash": self.config_hash,
            "scope_id": self.scope_id,
            "revision": self.revision,
            "last_event_time": self.last_event_time,
            "fast_state": self.fast_state.to_dict(),
            "mood_state": self.mood_state.to_dict(),
            "source_transition_id": self.source_transition_id,
        }

    @classmethod
    def initial(cls, scope_id: str, config: AffectConfig, timestamp: float = 0.0) -> AffectState:
        """Construct initial state for a scope at baseline."""
        return cls(
            schema_version=config.schema_version,
            config_version=config.config_version,
            config_hash=config.config_hash,
            scope_id=scope_id,
            revision=0,
            last_event_time=timestamp,
            fast_state=config.baseline,
            mood_state=config.baseline,
            source_transition_id=None,
        )


@dataclass(frozen=True, slots=True)
class AffectTransition:
    """Immutable record of one committed state transition for a turn."""

    transition_id: str
    scope_id: str
    revision: int
    turn_id: str
    idempotency_key: str
    prior_revision: int
    input_digest: str
    accepted_appraisal: dict[str, Any]
    pre_state: AffectVector
    after_state: AffectVector
    rendered_policy: ResponsePolicy
    outcome_disposition: str
    config_hash: str
    next_mood: AffectVector | None = None
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "scope_id": self.scope_id,
            "revision": self.revision,
            "turn_id": self.turn_id,
            "idempotency_key": self.idempotency_key,
            "prior_revision": self.prior_revision,
            "input_digest": self.input_digest,
            "accepted_appraisal": self.accepted_appraisal,
            "pre_state": self.pre_state.to_dict(),
            "after_state": self.after_state.to_dict(),
            "next_mood": self.next_mood.to_dict() if self.next_mood else None,
            "rendered_policy": self.rendered_policy.to_dict(),
            "outcome_disposition": self.outcome_disposition,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
        }
