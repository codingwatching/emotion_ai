"""Conversation-driven state changes, provenance, failure and negative controls."""

import asyncio
import hashlib
import json

import pytest

from aura_backend.affect.display import with_simulation_readouts
from aura_backend.affect.semantic import assess_interaction
from aura_backend.affect.service import AffectService
from aura_backend.providers.base import ProviderRequest, ProviderResult


def generator(events: list[dict[str, str]]):
    async def generate(request: ProviderRequest) -> ProviderResult:
        assert request.tools == () and request.session_id is None
        assert request.output_schema is not None
        assert request.disable_reasoning
        return ProviderResult(json.dumps({"events": events}))
    return generate


@pytest.mark.asyncio
async def test_meaning_changes_policy_before_reply_and_survives_restoration() -> None:
    service = AffectService()
    channels = []
    emotions = []
    # Ordinary language, none of the original exact trigger phrases.
    for index, (message, kind) in enumerate([
        ("How do stars form?", "exploration"),
        ("I'm delighted!", "celebration"),
        ("I'm frightened.", "distress"),
        ("I feel calm now.", "settling"),
    ]):
        prior, policy, events, appraisal, pre = await service.compute_provisional_policy(
            "scope", message, 1000 + index * 20,
            generate=generator([{"kind": kind, "evidence": message}]),
        )
        assert events == [f"conversation_{kind}"]
        assert appraisal.status == appraisal.analysis_status == "inferred"
        assert appraisal.source_sha256 == hashlib.sha256(message.encode()).hexdigest()
        assert appraisal.evidence_spans == (message,)
        assert policy.input_state == pre
        if kind == "exploration":
            assert pre.curiosity > prior.fast_state.curiosity
            assert policy.exploration == "exploratory"
        if kind == "distress":
            assert pre.affiliation >= prior.fast_state.affiliation
            assert policy.evidence_action == "proceed"
            assert policy.energy == "steady" and policy.exploration == "focused"
        if kind == "settling":
            assert pre.arousal < prior.fast_state.arousal
            assert pre.load < prior.fast_state.load
        state, transition = await service.stage_turn(
            "scope", str(index), str(index), str(index), prior, pre, policy,
            appraisal, None, 1000 + index * 20,
        )
        # Staging does not publish uncommitted changes.
        assert service.get_state("scope") == prior
        service.publish_turn("scope", state, transition)
        restored = AffectService()
        restored.set_state(state)
        assert restored.get_state("scope") == state
        readout = with_simulation_readouts({"post_state": state.fast_state.to_dict()})
        channels.append(json.dumps(readout["channels"], sort_keys=True))
        emotions.append(readout["display"]["emotion"]["name"])
    assert len(set(channels)) == 4
    assert emotions[0] == "Curious"
    assert "Excited" in emotions and "Concerned" in emotions
    assert service.get_state("other", 1000).revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [
    [{"kind": "exploration", "evidence": "invented"}],
    [{"kind": "verified_task_success", "evidence": "hello"}],
    [{"kind": "distress", "evidence": " "}],
    [{"kind": "distress", "evidence": "hello", "gain": "1"}],
])
async def test_invalid_proposal_cannot_move_state(events: list[dict[str, str]]) -> None:
    service = AffectService()
    prior, _policy, accepted, appraisal, pre = await service.compute_provisional_policy(
        "scope", "hello", 1000, generate=generator(events),
    )
    assert accepted == [] and pre == prior.fast_state
    assert appraisal.analysis_status == "invalid"


@pytest.mark.asyncio
async def test_abstention_and_failure_are_distinct_and_cancellation_propagates() -> None:
    result = await assess_interaction('She said "I am sad".', generator([]))
    assert result.status == "abstained" and result.events == ()

    async def unavailable(_request):
        raise RuntimeError("private provider detail")

    async def slow(_request):
        await asyncio.sleep(10)

    async def cancelled(_request):
        raise asyncio.CancelledError

    assert (await assess_interaction("hello", unavailable)).reason == "provider_failure"
    assert (await assess_interaction("hello", slow, timeout_seconds=0.001)).reason == "timeout"
    with pytest.raises(asyncio.CancelledError):
        await assess_interaction("hello", cancelled)


@pytest.mark.asyncio
async def test_lexical_and_semantic_matches_do_not_double_count_same_act() -> None:
    service = AffectService()
    _prior, _policy, events, _appraisal, pre = await service.compute_provisional_policy(
        "scope", "Thank you", 1000,
        generate=generator([{"kind": "affection", "evidence": "Thank you"}]),
    )
    assert events == ["explicit_collaboration"]
    assert pre.affiliation == pytest.approx(0.73)
    duplicate = await assess_interaction("hello", generator([
        {"kind": "distress", "evidence": "hello"},
        {"kind": "distress", "evidence": "hello"},
    ]))
    assert duplicate.events == (("conversation_distress", "hello"),)
