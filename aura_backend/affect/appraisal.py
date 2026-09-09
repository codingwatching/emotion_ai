"""Deterministic event appraisal and source-bound adapters for affective simulation."""

from __future__ import annotations

import re
from typing import Any

from aura_backend.affect.models import Appraisal

# Exact phrase patterns for deterministic event matching
_COLLABORATION_PATTERNS = [
    re.compile(r"\b(thank you|thanks|appreciate (your )?help|great work|good job|let's work together)\b", re.IGNORECASE),
]

_CORRECTION_PATTERNS = [
    re.compile(r"\b(actually|that's incorrect|that is incorrect|correction|you made a mistake|the correct answer is)\b", re.IGNORECASE),
]

_REPAIR_PATTERNS = [
    re.compile(r"\b(sorry about earlier|my apologies|let's start fresh|no hard feelings|excuse my frustration)\b", re.IGNORECASE),
]

_NEW_INFO_PATTERNS = [
    re.compile(r"\b(what if|have you considered|look at this new|did you know that|here is a puzzle|here's a puzzle)\b", re.IGNORECASE),
]


def appraise_user_message(
    message: str,
    *,
    task_facts: dict[str, Any] | None = None,
) -> list[str]:
    """Extract accepted starter event keys from user message and task facts.

    Strict negative controls:
    - User describing personal sadness or distress is NOT treated as hostility.
    - Blunt technical criticism is treated as a correction, never as contempt.
    - Ambiguous or unrecognized input returns empty list (zero impulse).
    """
    clean = message.strip()
    if not clean:
        return []

    accepted_events: list[str] = []

    # 1. Task facts have precedence if provided
    if task_facts:
        if task_facts.get("task_success") is True:
            accepted_events.append("verified_task_success")
        elif task_facts.get("task_failure") is True:
            accepted_events.append("verified_task_failure")

    # 2. Repair patterns
    for pat in _REPAIR_PATTERNS:
        if pat.search(clean):
            accepted_events.append("explicit_repair")
            break

    # 3. Source-supported correction (takes precedence over generic disagreement)
    is_quoted = bool(re.search(r'["\u201c\u201d]|\bsaid\b|\bwrote\b|\bquoting\b', clean, re.IGNORECASE))
    has_negation = bool(re.search(r'\b(not|never|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|no|neither)\b', clean[:50], re.IGNORECASE))
    has_sarcasm = bool(re.search(r'\b(yeah right|sure|totally|obviously|as if)\b', clean, re.IGNORECASE))

    if not is_quoted and not has_negation and not has_sarcasm:
        for pat in _CORRECTION_PATTERNS:
            if pat.search(clean):
                accepted_events.append("linguistic_correction_claim")
                break

    # 4. Collaboration / appreciation
    for pat in _COLLABORATION_PATTERNS:
        if pat.search(clean):
            accepted_events.append("explicit_collaboration")
            break

    # 5. New unresolved information / puzzle
    for pat in _NEW_INFO_PATTERNS:
        if pat.search(clean):
            accepted_events.append("new_unresolved_information")
            break

    return accepted_events


def build_appraisal_record(
    event_id: str,
    message: str,
    accepted_events: list[str],
    *,
    task_id: str | None = None,
) -> Appraisal:
    """Build a typed, provenance-bearing Appraisal record."""
    status = "observed" if accepted_events else "unknown"
    kind = accepted_events[0] if accepted_events else "neutral_conversation"

    return Appraisal(
        event_id=event_id,
        event_kind=kind,
        evidence_spans=(message[:120],) if message else (),
        status=status,
        task_id=task_id,
    )
