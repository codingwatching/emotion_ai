"""Controller displays survive classifier failure, replay and page reload."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aura_backend import main
from aura_backend.affect.display import with_simulation_readouts
from aura_backend.affect.models import AffectConfig, AffectState
from aura_backend.conversation.analysis import emotional_state_payload


def snapshot() -> dict:
    return {"revision": 4, "post_state": AffectConfig().baseline.to_dict()}


def test_unknown_tone_and_replay_use_same_published_controller_values() -> None:
    current = emotional_state_payload(None, simulation=snapshot())
    replay = emotional_state_payload(
        None, simulation={**snapshot(), "disposition": "replayed"}
    )
    assert current["name"] == "Unknown"
    assert current["brainwave"] == current["neurotransmitter"] == ""
    assert (
        current["simulation"]["display"]
        == replay["simulation"]["display"]
        == {
            "basis": "published_affect_state",
            "brainwave": "Alpha",
            "activation": 0.3,
            "dominant_channel": "serotonin_like",
        }
    )
    assert current["simulation"]["channels"]["serotonin_like"] == 0.75


@pytest.mark.parametrize(
    "activation,band",
    [(0, "Delta"), (0.2, "Theta"), (0.3, "Alpha"), (0.6, "Beta"), (0.9, "Gamma")],
)
def test_rhythm_tracks_activation_and_channels_track_controller_changes(
    activation: float, band: str
) -> None:
    sim = snapshot()
    sim["post_state"]["arousal"] = activation
    sim["post_state"]["load"] = 0.95
    result = with_simulation_readouts(sim)
    assert result["display"]["brainwave"] == band
    assert result["display"]["dominant_channel"] == "cortisol_like"
    assert result["channels"]["cortisol_like"] == 0.95


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"arousal": 0.3},
        {**AffectConfig().baseline.to_dict(), "arousal": float("nan")},
        {**AffectConfig().baseline.to_dict(), "load": 2},
    ],
)
def test_missing_or_malformed_state_does_not_create_display_values(
    value: object,
) -> None:
    result = with_simulation_readouts(
        {"post_state": value, "display": {"brainwave": "fake"}, "channels": {}}
    )
    assert "display" not in result and "channels" not in result


def test_saved_simulation_endpoint_reads_exact_user_without_provider_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = replace(AffectState.initial("Ty", AffectConfig()), revision=4)
    reads = []

    def get_head(user: str) -> AffectState | None:
        reads.append(user)
        return state if user == "Ty" else None

    monkeypatch.setattr(
        main,
        "storage_boundary",
        SimpleNamespace(repository=SimpleNamespace(get_affect_head=get_head)),
    )
    client = TestClient(
        main.create_app()
    )  # No lifespan/provider startup needed for this read seam.
    response = client.get("/simulation/Ty")
    assert response.status_code == 200
    sim = response.json()["simulation"]
    assert sim["revision"] == 4 and sim["disposition"] == "restored"
    assert sim["display"]["brainwave"] == "Alpha"
    assert client.get("/simulation/new-user").json() == {"simulation": None}
    assert reads == ["Ty", "new-user"]
