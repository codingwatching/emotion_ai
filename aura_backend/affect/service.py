"""Runtime service for per-scope affective simulation sequencing."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from typing import Any

from aura_backend.affect.appraisal import appraise_user_message, build_appraisal_record
from aura_backend.affect.dynamics import (
    apply_observed_outcome,
    apply_pre_state,
    calculate_turn_impulse,
    compute_decay,
    compute_next_mood,
)
from aura_backend.affect.models import (
    AffectConfig,
    AffectState,
    AffectTransition,
    AffectVector,
    Appraisal,
    ResponsePolicy,
    TaskOutcome,
)
from aura_backend.affect.policy import render_policy


class AffectService:
    """Per-scope affective simulation sequencer with serialized updates."""

    def __init__(
        self,
        config: AffectConfig | None = None,
        repository: Any | None = None,
    ) -> None:
        self.config = config or AffectConfig()
        self.repository = repository
        self._states: dict[str, AffectState] = {}
        self._transitions: dict[str, list[AffectTransition]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, scope_id: str) -> asyncio.Lock:
        if scope_id not in self._locks:
            self._locks[scope_id] = asyncio.Lock()
        return self._locks[scope_id]

    def get_state(self, scope_id: str, timestamp: float | None = None) -> AffectState:
        """Get or initialize the current state for a scope."""
        if scope_id not in self._states:
            if self.repository is not None and hasattr(self.repository, "get_affect_head"):
                persisted = self.repository.get_affect_head(scope_id)
                if persisted is not None:
                    self._states[scope_id] = persisted
                    return persisted
            ts = timestamp if timestamp is not None else time.time()
            self._states[scope_id] = AffectState.initial(scope_id, self.config, ts)
        return self._states[scope_id]

    def set_state(self, state: AffectState) -> None:
        """Explicitly set the current state (e.g. upon restore from storage)."""
        self._states[state.scope_id] = state

    async def compute_provisional_policy(
        self,
        scope_id: str,
        message: str,
        timestamp: float,
        *,
        task_facts: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> tuple[AffectState, ResponsePolicy, list[str], Appraisal, AffectVector]:
        """Compute provisional pre-response state and policy under per-scope lock.

        Returns:
            (prior_state, rendered_policy, accepted_events, appraisal, pre_state)
        """
        async with self._get_lock(scope_id):
            prior_state = self.get_state(scope_id, timestamp)
            dt = max(0.0, timestamp - prior_state.last_event_time)

            mood_decayed, _target, fast_decayed = compute_decay(
                prior_state.fast_state,
                prior_state.mood_state,
                dt,
                self.config,
            )
            accepted_events = appraise_user_message(message, task_facts=task_facts)
            turn_impulse = calculate_turn_impulse(accepted_events, self.config)
            pre_state = apply_pre_state(fast_decayed, turn_impulse)

            eid = event_id or uuid.uuid4().hex
            appraisal = build_appraisal_record(
                event_id=eid,
                message=message,
                accepted_events=accepted_events,
            )
            policy = render_policy(pre_state)

            return prior_state, policy, accepted_events, appraisal, pre_state

    async def commit_turn(
        self,
        scope_id: str,
        turn_id: str,
        idempotency_key: str,
        input_digest: str,
        prior_state: AffectState,
        pre_state: AffectVector,
        policy: ResponsePolicy,
        appraisal: Appraisal,
        outcome: TaskOutcome | None,
        timestamp: float,
    ) -> tuple[AffectState, AffectTransition]:
        """Commit a complete turn, applying outcomes and updating head state."""
        async with self._get_lock(scope_id):
            dt = max(0.0, timestamp - prior_state.last_event_time)
            mood_decayed, _, _ = compute_decay(
                prior_state.fast_state,
                prior_state.mood_state,
                dt,
                self.config,
            )
            after_state, disposition = apply_observed_outcome(pre_state, outcome, self.config)
            next_mood = compute_next_mood(mood_decayed, after_state, self.config)

            next_revision = prior_state.revision + 1
            digest_seed = f"{scope_id}:{next_revision}:{turn_id}:{idempotency_key}".encode("utf-8")
            id_suffix = hashlib.sha256(digest_seed).hexdigest()[:8]
            transition_id = f"trans_{scope_id}_{next_revision}_{id_suffix}"

            transition = AffectTransition(
                transition_id=transition_id,
                scope_id=scope_id,
                revision=next_revision,
                turn_id=turn_id,
                idempotency_key=idempotency_key,
                prior_revision=prior_state.revision,
                input_digest=input_digest,
                accepted_appraisal=appraisal.to_dict(),
                pre_state=pre_state,
                after_state=after_state,
                rendered_policy=policy,
                outcome_disposition=disposition,
                config_hash=self.config.config_hash,
                next_mood=next_mood,
                timestamp=timestamp,
            )

            new_state = AffectState(
                schema_version=self.config.schema_version,
                config_version=self.config.config_version,
                config_hash=self.config.config_hash,
                scope_id=scope_id,
                revision=next_revision,
                last_event_time=timestamp,
                fast_state=after_state,
                mood_state=next_mood,
                source_transition_id=transition_id,
            )

            self._states[scope_id] = new_state
            if scope_id not in self._transitions:
                self._transitions[scope_id] = []
            self._transitions[scope_id].append(transition)

            return new_state, transition
