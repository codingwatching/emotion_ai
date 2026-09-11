"""Runtime service for per-scope affective simulation sequencing."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import asdict
from typing import Any

from aura_backend.affect.appraisal import appraise_user_message, build_appraisal_record
from aura_backend.affect.regulation import classify_events, regulate_interpretations
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


class AsyncReentrantLock:
    """An asyncio reentrant lock tracking current task and recursion depth."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._count = 0

    async def acquire(self) -> bool:
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("AsyncReentrantLock must be used within an asyncio task")
        if self._owner == current_task:
            self._count += 1
            return True
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1
        return True

    def release(self) -> None:
        current_task = asyncio.current_task()
        if self._owner != current_task:
            raise RuntimeError("Cannot release a lock owned by another task")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self) -> AsyncReentrantLock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


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
        self._locks: dict[str, AsyncReentrantLock] = {}
        # Staged regulation decisions: populated in compute_provisional_policy,
        # consumed in stage_turn, cleared in discard_staged_turn.
        from aura_backend.affect.regulation import RegulationDecision
        self._staged_decisions: dict[str, RegulationDecision] = {}


    def _get_lock(self, scope_id: str) -> AsyncReentrantLock:
        if scope_id not in self._locks:
            self._locks[scope_id] = AsyncReentrantLock()
        return self._locks[scope_id]

    def scope_lock(self, scope_id: str) -> AsyncReentrantLock:
        """Get the re-entrant per-scope lock for serializing an entire turn."""
        return self._get_lock(scope_id)

    def get_state(self, scope_id: str, timestamp: float | None = None) -> AffectState:
        """Get or initialize the current state for a scope."""
        if self.repository is not None and hasattr(self.repository, "get_affect_head"):
            persisted = self.repository.get_affect_head(scope_id)
            if persisted is not None:
                current = self._states.get(scope_id)
                if current != persisted:
                    # SQLite is authority, including equal-revision identity
                    # mismatches left by older runtime versions.
                    self._states[scope_id] = persisted
                    return persisted
        if scope_id not in self._states:
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

            # Apply non-punitive regulation, passing the pre-impulse decayed state so
            # the regulator can revert affiliation for disrespect events.
            interpretations = classify_events(message, eid, accepted_events, task_facts)
            non_grievance_state = apply_pre_state(
                fast_decayed, calculate_turn_impulse(
                    [event for event in accepted_events if event != "repeated_directed_contempt"], self.config,
                ),
            )
            regulation_result = regulate_interpretations(
                pre_state, interpretations, self.config, non_grievance_state,
            )
            pre_state = regulation_result.regulated_state
            # Stage the decision for persistence in stage_turn
            self._staged_decisions[scope_id] = regulation_result

            appraisal = build_appraisal_record(
                event_id=eid,
                message=message,
                accepted_events=accepted_events,
            )
            policy = render_policy(pre_state, regulation=regulation_result)

            return prior_state, policy, accepted_events, appraisal, pre_state


    async def stage_turn(
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
        """Stage a turn transition without publishing to in-memory head state."""
        async with self._get_lock(scope_id):
            dt = max(0.0, timestamp - prior_state.last_event_time)
            mood_decayed, _, fast_decayed = compute_decay(
                prior_state.fast_state,
                prior_state.mood_state,
                dt,
                self.config,
            )
            regulation_decision = self._staged_decisions.get(scope_id)
            already_appraised = (
                outcome is not None and outcome.success is not None
                and regulation_decision is not None
                and any(
                    item.claim_kind == ("verified_success" if outcome.success else "verified_failure")
                    and item.validation_status == "verified"
                    for item in regulation_decision.accepted_interpretations
                )
            )
            # The legacy seam describes one outcome per turn, not independent
            # repeated rewards. Do not apply that same observation twice.
            after_state, disposition = apply_observed_outcome(pre_state, None if already_appraised else outcome, self.config)
            if already_appraised and outcome is not None:
                disposition = "success" if outcome.success else "failure"
            decay_values = asdict(fast_decayed)
            capped_values = {}
            for dimension, value in asdict(after_state).items():
                delta = value - decay_values[dimension]
                cap = self.config.max_turn_impulse
                capped_values[dimension] = value if abs(delta) <= cap else decay_values[dimension] + max(-cap, min(cap, delta))
            after_state = AffectVector.from_dict(capped_values).clip()
            next_mood = compute_next_mood(mood_decayed, after_state, self.config)

            next_revision = prior_state.revision + 1
            digest_seed = f"{scope_id}:{next_revision}:{turn_id}:{idempotency_key}".encode("utf-8")
            id_suffix = hashlib.sha256(digest_seed).hexdigest()[:8]
            transition_id = f"trans_{scope_id}_{next_revision}_{id_suffix}"

            # Merge staged regulation decision into appraisal provenance
            regulation_decision = self._staged_decisions.pop(scope_id, None)
            appraisal_dict = appraisal.to_dict()
            appraisal_dict["outcome_application"] = "already_appraised" if already_appraised else "post_response" if outcome is not None else "unavailable"
            appraisal_dict["task_outcome"] = outcome.to_dict() if outcome is not None else None
            if regulation_decision is not None:
                appraisal_dict["regulation_decision"] = regulation_decision.to_dict()

            transition = AffectTransition(
                transition_id=transition_id,
                scope_id=scope_id,
                revision=next_revision,
                turn_id=turn_id,
                idempotency_key=idempotency_key,
                prior_revision=prior_state.revision,
                input_digest=input_digest,
                accepted_appraisal=appraisal_dict,
                pre_state=pre_state,
                after_state=after_state,
                rendered_policy=policy,
                outcome_disposition=disposition,
                config_hash=self.config.config_hash,
                next_mood=next_mood,
                timestamp=timestamp,
            )

            staged_state = AffectState(
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

            return staged_state, transition

    def publish_turn(
        self,
        scope_id: str,
        new_state: AffectState,
        transition: AffectTransition,
    ) -> None:
        """Publish a staged turn to in-memory state after verified persistence."""
        self._states[scope_id] = new_state
        if scope_id not in self._transitions:
            self._transitions[scope_id] = []
        self._transitions[scope_id].append(transition)

    def discard_staged_turn(self, scope_id: str) -> None:
        """Cleanly discard staged turn and re-sync memory from persistent repository if available."""
        # Clear any staged regulation decision so it is not applied to the next turn
        self._staged_decisions.pop(scope_id, None)
        if self.repository is not None and hasattr(self.repository, "get_affect_head"):
            head = self.repository.get_affect_head(scope_id)
            if head is not None:
                self._states[scope_id] = head


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
        """Commit a complete turn (stage and publish immediately in memory)."""
        new_state, transition = await self.stage_turn(
            scope_id=scope_id,
            turn_id=turn_id,
            idempotency_key=idempotency_key,
            input_digest=input_digest,
            prior_state=prior_state,
            pre_state=pre_state,
            policy=policy,
            appraisal=appraisal,
            outcome=outcome,
            timestamp=timestamp,
        )
        self.publish_turn(scope_id, new_state, transition)
        return new_state, transition
