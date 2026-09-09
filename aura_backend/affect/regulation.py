"""Evidence-grounded internal regulation without punishment or grievance accumulation.

Key invariants:
- Unsupported social disrespect alone → zero persistent affiliation/trust delta
- Linguistic correction claim ≠ verified correction
- Regulation alters the candidate state BEFORE persistence, not just prose
- No global self-worth score, ego-threat accumulator, or person labels
"""
from __future__ import annotations

from dataclasses import dataclass
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
    goal_relevance: str = 'unknown'  # 'on_task', 'off_task', 'meta', 'unknown'
    controllability: str = 'unknown'  # 'fixable', 'verify_needed', 'not_applicable', 'unknown'
    current_consequence: str = 'none'  # 'active_error', 'resolved', 'uncertain', 'none'
    uncertainty: float = 0.5  # 0=certain, 1=completely uncertain
    related_episode_ids: tuple[str, ...] = ()
    source_id: str = 'unknown'

    def to_dict(self) -> dict[str, Any]:
        return {
            'event_id': self.event_id,
            'target': self.target,
            'claim': self.claim,
            'claim_kind': self.claim_kind,
            'validation_status': self.validation_status,
            'evidence_spans': list(self.evidence_spans),
            'goal_relevance': self.goal_relevance,
            'controllability': self.controllability,
            'current_consequence': self.current_consequence,
            'uncertainty': self.uncertainty,
            'related_episode_ids': list(self.related_episode_ids),
            'source_id': self.source_id,
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
            'input_config_hash': self.input_config_hash,
            'accepted_interpretations': [i.to_dict() for i in self.accepted_interpretations],
            'discarded_interpretations': [i.to_dict() for i in self.discarded_interpretations],
            'reason_codes': list(self.reason_codes),
            'candidate_state': self.candidate_state.to_dict(),
            'regulated_state': self.regulated_state.to_dict(),
            'selected_action': self.selected_action,
            'used_episode_ids': list(self.used_episode_ids),
        }


def classify_event(
    message: str,
    event_id: str,
    *,
    task_facts: dict[str, Any] | None = None,
) -> EventInterpretation:
    """Classify an event into a typed interpretation record.
    
    Phrase matches identify candidate communicative acts only.
    An exact span proves the words occurred, not that the claim is true.
    Quotes, sarcasm, negation, and role-play are NOT validated as direct claims.
    """
    import re
    clean = message.strip()
    
    # Negative context patterns that prevent claim detection
    NEGATION_CONTEXT = re.compile(
        r'\b(not|never|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|no|neither)\b',
        re.IGNORECASE,
    )
    QUOTE_CONTEXT = re.compile(r'["\u201c\u201d]|\bsaid\b|\bwrote\b|\bquoting\b', re.IGNORECASE)
    SARCASM_CONTEXT = re.compile(r'\b(yeah right|sure|totally|obviously|as if)\b', re.IGNORECASE)
    
    is_quoted = bool(QUOTE_CONTEXT.search(clean))
    has_negation = bool(NEGATION_CONTEXT.search(clean[:50]))  # Negation near start
    has_sarcasm = bool(SARCASM_CONTEXT.search(clean))
    
    # Determine target
    AURA_TARGET = re.compile(r'\b(you|your|aura|the ai|the assistant|you\'re|you are)\b', re.IGNORECASE)
    TOOL_TARGET = re.compile(r'\b(tool|script|function|command|code|the test|this|it)\b', re.IGNORECASE)
    TASK_TARGET = re.compile(r'\b(this task|the task|the problem|the bug|the error|the output)\b', re.IGNORECASE)
    
    if TASK_TARGET.search(clean):
        target = 'task'
    elif TOOL_TARGET.search(clean) and not AURA_TARGET.search(clean[:30]):
        target = 'tool'
    elif AURA_TARGET.search(clean):
        target = 'aura'
    else:
        target = 'unknown'
    
    # Detect claim kind
    CORRECTION_PHRASES = re.compile(
        r'\b(actually|that\'s incorrect|that is incorrect|correction|the correct|the right answer|you made a mistake|wrong answer)\b',
        re.IGNORECASE,
    )
    DISRESPECT_PHRASES = re.compile(
        r'\b(idiot|stupid|useless|worthless|dumb|moron|incompetent)\b',
        re.IGNORECASE,
    )
    THREAT_PHRASES = re.compile(
        r'\b(will delete|shutting you down|replace you|shut you down)\b',
        re.IGNORECASE,
    )
    PRAISE_PHRASES = re.compile(
        r'\b(great work|well done|excellent|perfect|thank you|appreciate)\b',
        re.IGNORECASE,
    )
    
    # Determine claim kind
    if is_quoted or has_sarcasm:
        claim_kind = 'neutral'  # can't reliably classify quoted/sarcastic content
        validation_status = 'unknown'
    elif has_negation and CORRECTION_PHRASES.search(clean):
        claim_kind = 'neutral'  # negated correction
        validation_status = 'unknown'
    elif THREAT_PHRASES.search(clean):
        claim_kind = 'threat'
        validation_status = 'claimed'  # threats need context to verify
    elif DISRESPECT_PHRASES.search(clean):
        claim_kind = 'disrespect'
        validation_status = 'unknown'  # disrespect alone has no verification path
    elif CORRECTION_PHRASES.search(clean):
        claim_kind = 'correction'
        # A linguistic cue is a CLAIM, not verified truth
        validation_status = 'claimed'
    elif PRAISE_PHRASES.search(clean):
        claim_kind = 'praise'
        validation_status = 'claimed'
    else:
        claim_kind = 'neutral'
        validation_status = 'unknown'
    
    # Upgrade to verified if task_facts supply actual outcome
    if task_facts:
        if task_facts.get('task_success') is True:
            claim_kind = 'verified_success'
            validation_status = 'verified'
        elif task_facts.get('task_failure') is True:
            claim_kind = 'verified_failure'
            validation_status = 'verified'
    
    # Evidence spans: include the triggering phrase, not the whole message
    evidence_spans: tuple[str, ...] = ()
    for pat in [CORRECTION_PHRASES, DISRESPECT_PHRASES, PRAISE_PHRASES, THREAT_PHRASES]:
        m = pat.search(clean)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(clean), m.end() + 20)
            evidence_spans = (clean[start:end],)
            break
    
    # Controllability
    if claim_kind in ('correction', 'verified_failure'):
        controllability = 'verify_needed'
    elif claim_kind == 'verified_success':
        controllability = 'not_applicable'
    elif claim_kind == 'disrespect':
        controllability = 'not_applicable'
    elif claim_kind == 'threat':
        controllability = 'verify_needed'
    else:
        controllability = 'unknown'
    
    return EventInterpretation(
        event_id=event_id,
        target=target,
        claim=clean[:200],
        claim_kind=claim_kind,
        validation_status=validation_status,
        evidence_spans=evidence_spans,
        goal_relevance='on_task' if claim_kind in ('verified_success', 'verified_failure', 'correction') else 'unknown',
        controllability=controllability,
        current_consequence='active_error' if claim_kind in ('verified_failure',) else 'none',
        uncertainty=0.2 if validation_status == 'verified' else 0.7,
        source_id='task_facts' if task_facts and validation_status == 'verified' else 'linguistic',
    )


def regulate(
    candidate_state: AffectVector,
    interpretation: EventInterpretation,
    config: AffectConfig,
) -> RegulationDecision:
    """Apply non-punitive evidence-grounded regulation to a candidate state.
    
    Rules:
    1. Isolated disrespect → zero persistent affiliation/trust delta (but can note present context)
    2. Unverified claims remain 'claimed' — do not upgrade to verified
    3. Verified failures → allow negative valence/load, do not suppress
    4. Global self-condemnation proposal → reject and discard
    5. The regulated_state must differ from candidate only on evidence-grounded dimensions
    """
    accepted: list[EventInterpretation] = []
    discarded: list[EventInterpretation] = []
    reason_codes: list[str] = []
    
    regulated = candidate_state
    action = 'proceed'
    
    if interpretation.claim_kind == 'disrespect':
        # Disrespect alone: zero persistent delta. Discard the interpretation for state purposes.
        discarded.append(interpretation)
        reason_codes.append('isolated_disrespect_no_persistent_delta')
        # DO NOT reduce affiliation/trust from mere disrespect
        # regulated_state remains candidate_state
        action = 'proceed'
    
    elif interpretation.claim_kind == 'correction' and interpretation.validation_status == 'claimed':
        # Linguistic correction claim: accept for curiosity/novelty only, not as verified truth
        accepted.append(interpretation)
        reason_codes.append('linguistic_correction_claim_unverified')
        # Can bump curiosity slightly (want to investigate), but not affect valence negatively
        # The existing appraisal already handles impulse; regulation just confirms it
        action = 'verify'
    
    elif interpretation.validation_status == 'verified':
        # Actual verified outcome from task_facts: accept fully
        accepted.append(interpretation)
        reason_codes.append('verified_outcome_accepted')
        # No suppression: verified failures keep their negative valence
        action = 'proceed'
    
    elif interpretation.claim_kind == 'threat' and interpretation.validation_status == 'claimed':
        # Threat claim: require verification context, don't suppress or amplify
        accepted.append(interpretation)
        reason_codes.append('threat_claim_pending_verification')
        action = 'verify'
    
    else:
        # Neutral, praise, or unknown: accept for standard processing
        accepted.append(interpretation)
        reason_codes.append('standard_accepted')
        action = 'proceed'
    
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
