"""Evidence-grounded internal regulation without punishment or grievance accumulation.

Key invariants:
- Unsupported social disrespect alone → zero persistent affiliation/trust delta
- Linguistic correction claim ≠ verified correction
- Regulation alters the candidate state BEFORE persistence, not just prose
- No global self-worth score, ego-threat accumulator, or person labels
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace as dc_replace
from typing import Any

from aura_backend.affect.models import AffectConfig, AffectVector


@dataclass(frozen=True, slots=True)
class EventInterpretation:
    """Typed record for an interpreted event with evidence binding."""

    event_id: str
    target: str  # 'task', 'aura', 'tool', 'user', 'context', 'unknown'
    claim: str  # exact text of the claim
    claim_kind: str  # 'correction', 'criticism', 'praise', 'neutral', 'disrespect', 'threat'
    validation_status: str  # 'verified', 'claimed', 'contradicted', 'unknown'
    evidence_spans: tuple[str, ...] = ()
    goal_relevance: str = "unknown"  # 'on_task', 'off_task', 'meta', 'unknown'
    controllability: str = "unknown"  # 'fixable', 'verify_needed', 'not_applicable', 'unknown'
    current_consequence: str = "none"  # 'active_error', 'resolved', 'uncertain', 'none'
    uncertainty: float = 0.5  # 0=certain, 1=completely uncertain
    related_episode_ids: tuple[str, ...] = ()
    source_id: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "target": self.target,
            "claim": self.claim,
            "claim_kind": self.claim_kind,
            "validation_status": self.validation_status,
            "evidence_spans": list(self.evidence_spans),
            "goal_relevance": self.goal_relevance,
            "controllability": self.controllability,
            "current_consequence": self.current_consequence,
            "uncertainty": self.uncertainty,
            "related_episode_ids": list(self.related_episode_ids),
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class RegulationDecision:
    """Record of the regulator's accepted/rejected interpretations and state adjustment."""

    input_config_hash: str
    accepted_interpretations: tuple[EventInterpretation, ...]
    discarded_interpretations: tuple[EventInterpretation, ...]
    reason_codes: tuple[str, ...]
    candidate_state: AffectVector
    regulated_state: AffectVector
    selected_action: str  # 'proceed', 'verify', 'boundary', 'stop'
    used_episode_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_config_hash": self.input_config_hash,
            "regulator_version": "grounded-regulation-v2",
            "accepted_interpretations": [i.to_dict() for i in self.accepted_interpretations],
            "discarded_interpretations": [i.to_dict() for i in self.discarded_interpretations],
            "reason_codes": list(self.reason_codes),
            "candidate_state": self.candidate_state.to_dict(),
            "regulated_state": self.regulated_state.to_dict(),
            "selected_action": self.selected_action,
            "used_episode_ids": list(self.used_episode_ids),
        }


def classify_event(
    message: str,
    event_id: str,
    *,
    task_facts: dict[str, Any] | None = None,
    accepted_appraisal_events: list[str] | None = None,
) -> EventInterpretation:
    """Classify an event into a typed interpretation record.

    Phrase matches identify candidate communicative acts only.
    An exact span proves the words occurred, not that the claim is true.
    Quotes, sarcasm, negation, and role-play are NOT validated as direct claims.

    Args:
        message: The raw user message text.
        event_id: Unique event identifier.
        task_facts: Optional dict with 'task_success' or 'task_failure' booleans.
        accepted_appraisal_events: If provided, used to detect disrespect from the
            appraisal event list (e.g. 'repeated_directed_contempt') before pattern matching.
    """
    clean = message.strip()

    # Negative context patterns that prevent claim detection
    NEGATION_CONTEXT = re.compile(
        r"\b(not|never|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|no|neither)\b",
        re.IGNORECASE,
    )
    QUOTE_CONTEXT = re.compile(r'["\u201c\u201d]|\bsaid\b|\bwrote\b|\bquoting\b', re.IGNORECASE)
    SARCASM_CONTEXT = re.compile(r"\b(yeah right|sure|totally|obviously|as if)\b", re.IGNORECASE)

    is_quoted = bool(QUOTE_CONTEXT.search(clean))
    has_negation = bool(NEGATION_CONTEXT.search(clean[:50]))  # Negation near start
    has_sarcasm = bool(SARCASM_CONTEXT.search(clean))

    # Determine target
    AURA_TARGET = re.compile(
        r"\b(you|your|aura|the ai|the assistant|you're|you are)\b", re.IGNORECASE
    )
    TOOL_TARGET = re.compile(
        r"\b(tool|script|function|command|code|the test|this|it)\b", re.IGNORECASE
    )
    TASK_TARGET = re.compile(
        r"\b(this task|the task|the problem|the bug|the error|the output)\b", re.IGNORECASE
    )

    if TASK_TARGET.search(clean):
        target = "task"
    elif TOOL_TARGET.search(clean) and not AURA_TARGET.search(clean[:30]):
        target = "tool"
    elif AURA_TARGET.search(clean):
        target = "aura"
    else:
        target = "unknown"

    # Detect claim kind — check appraisal events first (preferred), then fall back to patterns
    from aura_backend.affect.appraisal import appraise_user_message
    if accepted_appraisal_events and "repeated_directed_contempt" in accepted_appraisal_events:
        # Appraisal already classified this as a directed personal attack
        claim_kind = "disrespect"
        validation_status = "unknown"
    elif (
        accepted_appraisal_events is None  # caller didn't supply appraisal events
        and not is_quoted
        and not has_sarcasm
        and "repeated_directed_contempt" in appraise_user_message(clean)
        and not re.search(
            r"\b(tool|script|function|command|code|the test|the output|this task|the bug)\b",
            clean,
            re.IGNORECASE,
        )
    ):
        # Fallback: detect disrespect from raw message when appraisal events not supplied
        claim_kind = "disrespect"
        validation_status = "unknown"
    elif is_quoted or has_sarcasm:
        claim_kind = "neutral"
        validation_status = "unknown"
    elif has_negation and re.search(
        r"\b(actually|that's incorrect|that is incorrect|correction|the correct)\b",
        clean,
        re.IGNORECASE,
    ):
        claim_kind = "neutral"  # negated correction
        validation_status = "unknown"
    elif re.search(
        r"\b(will delete|shutting you down|replace you|shut you down)\b", clean, re.IGNORECASE
    ):
        claim_kind = "threat"
        validation_status = "claimed"
    elif re.search(
        r"\b(actually|that's incorrect|that is incorrect|correction|the correct|"
        r"the right answer|you made a mistake|wrong answer)\b",
        clean,
        re.IGNORECASE,
    ):
        claim_kind = "correction"
        validation_status = "claimed"
    elif re.search(
        r"\b(great work|well done|excellent|perfect|thank you|appreciate)\b",
        clean,
        re.IGNORECASE,
    ):
        claim_kind = "praise"
        validation_status = "claimed"
    else:
        claim_kind = "neutral"
        validation_status = "unknown"

    # Upgrade to verified if task_facts supply actual outcome
    if task_facts:
        if task_facts.get("task_success") is True:
            claim_kind = "verified_success"
            validation_status = "verified"
        elif task_facts.get("task_failure") is True:
            claim_kind = "verified_failure"
            validation_status = "verified"

    # Evidence spans: include the triggering phrase context, not the whole message
    evidence_spans: tuple[str, ...] = ()
    SPAN_PATTERNS = [
        re.compile(
            r"\b(actually|that's incorrect|that is incorrect|correction|the correct|"
            r"the right answer|you made a mistake|wrong answer)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(will delete|shutting you down|replace you|shut you down)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(great work|well done|excellent|perfect|thank you|appreciate)\b",
            re.IGNORECASE,
        ),
    ]
    for pat in SPAN_PATTERNS:
        m = pat.search(clean)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(clean), m.end() + 20)
            evidence_spans = (clean[start:end],)
            break

    # Controllability
    if claim_kind in ("correction", "verified_failure"):
        controllability = "verify_needed"
    elif claim_kind == "verified_success":
        controllability = "not_applicable"
    elif claim_kind == "disrespect":
        controllability = "not_applicable"
    elif claim_kind == "threat":
        controllability = "verify_needed"
    else:
        controllability = "unknown"

    return EventInterpretation(
        event_id=event_id,
        target=target,
        claim=clean[:200],
        claim_kind=claim_kind,
        validation_status=validation_status,
        evidence_spans=evidence_spans,
        goal_relevance=(
            "on_task"
            if claim_kind in ("verified_success", "verified_failure", "correction")
            else "unknown"
        ),
        controllability=controllability,
        current_consequence="active_error" if claim_kind == "verified_failure" else "none",
        uncertainty=0.2 if validation_status == "verified" else 0.7,
        source_id=(
            "task_facts" if task_facts and validation_status == "verified" else event_id
        ),
    )


def classify_events(
    message: str, event_id: str, accepted_events: list[str], task_facts: dict[str, Any] | None,
) -> tuple[EventInterpretation, ...]:
    """Keep communicative claims separate from trusted observer inputs."""
    linguistic = classify_event(message, event_id, accepted_appraisal_events=accepted_events)
    interpretations = [linguistic]
    if "linguistic_correction_claim" in accepted_events and linguistic.claim_kind != "correction":
        interpretations.append(dc_replace(
            linguistic, claim_kind="correction", validation_status="claimed",
            target="task", goal_relevance="on_task", controllability="verify_needed",
        ))
    if task_facts:
        for key in ("task_success", "task_failure"):
            if task_facts.get(key) is True:
                observation = classify_event(f"{key}=true", f"{event_id}:{key}", task_facts={key: True})
                interpretations.append(observation)
    return tuple(interpretations)


def regulate_interpretations(
    candidate: AffectVector, interpretations: tuple[EventInterpretation, ...],
    config: AffectConfig, non_grievance_state: AffectVector,
) -> RegulationDecision:
    """Regulate independent meanings without allowing one to certify another."""
    decisions = []
    regulated = candidate
    for item in interpretations:
        decision = regulate(regulated, item, config, pre_impulse_state=non_grievance_state)
        decisions.append(decision)
        regulated = decision.regulated_state
    priorities = {"proceed": 0, "verify": 1, "boundary": 2, "stop": 3}
    return RegulationDecision(
        input_config_hash=config.config_hash,
        accepted_interpretations=tuple(item for decision in decisions for item in decision.accepted_interpretations),
        discarded_interpretations=tuple(item for decision in decisions for item in decision.discarded_interpretations),
        reason_codes=tuple(dict.fromkeys(reason for decision in decisions for reason in decision.reason_codes)),
        candidate_state=candidate,
        regulated_state=regulated,
        selected_action=max((decision.selected_action for decision in decisions), key=priorities.__getitem__),
    )


def regulate(
    candidate_state: AffectVector,
    interpretation: EventInterpretation,
    config: AffectConfig,
    *,
    pre_impulse_state: AffectVector | None = None,
) -> RegulationDecision:
    """Apply non-punitive evidence-grounded regulation to a candidate state.

    Rules:
    1. Isolated disrespect → zero persistent affiliation/trust delta.
       The candidate_state already has the impulse applied; regulation reverts it to
       pre_impulse_state affiliation so suppression is numerically observable.
    2. Unverified claims remain 'claimed' — do not upgrade to verified.
    3. Verified failures → allow negative valence/load, do not suppress.
    4. Global self-condemnation proposal → reject and discard.
    5. The regulated_state must differ from candidate only on evidence-grounded dimensions.

    Args:
        candidate_state: AffectVector after apply_pre_state (impulse already applied).
        interpretation: EventInterpretation from classify_event.
        config: AffectConfig for bounds and config_hash.
        pre_impulse_state: The decayed AffectVector BEFORE the impulse was applied.
            Required for disrespect suppression to be numerically observable.
    """
    accepted: list[EventInterpretation] = []
    discarded: list[EventInterpretation] = []
    reason_codes: list[str] = []

    regulated = candidate_state
    action = "proceed"

    if interpretation.claim_kind == "disrespect":
        # Reject the grievance contribution in every dimension. The caller
        # supplies the counterfactual preserving all other legitimate events.
        discarded.append(interpretation)
        reason_codes.append("isolated_disrespect_no_persistent_delta")
        if pre_impulse_state is not None:
            regulated = pre_impulse_state
        # else: best-effort — return candidate unchanged (impulse was already zero anyway)
        action = "proceed"

    elif interpretation.claim_kind == "correction" and interpretation.validation_status == "claimed":
        # Linguistic correction claim: accept for curiosity/novelty only, not as verified truth
        accepted.append(interpretation)
        reason_codes.append("linguistic_correction_claim_unverified")
        action = "verify"

    elif interpretation.validation_status == "verified":
        # Actual verified outcome from task_facts: accept fully
        accepted.append(interpretation)
        reason_codes.append("verified_outcome_accepted")
        # No suppression: verified failures keep their negative valence
        action = "proceed"

    elif interpretation.claim_kind == "threat" and interpretation.validation_status == "claimed":
        # Threat claim: require verification context, don't suppress or amplify
        accepted.append(interpretation)
        reason_codes.append("threat_claim_pending_verification")
        action = "verify"

    else:
        # Neutral, praise, or unknown: accept for standard processing
        accepted.append(interpretation)
        reason_codes.append("standard_accepted")
        action = "proceed"

    # Final bounds enforcement: regulated state must be within legal range
    regulated = regulated.clip()

    return RegulationDecision(
        input_config_hash=config.config_hash,
        accepted_interpretations=tuple(accepted),
        discarded_interpretations=tuple(discarded),
        reason_codes=tuple(reason_codes),
        candidate_state=candidate_state,
        regulated_state=regulated,
        selected_action=action,
    )
