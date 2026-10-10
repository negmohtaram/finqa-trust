import socket
from dataclasses import dataclass

import pytest

from finqa_trust.model import (ModelRequest, ModelResponse, RecordingClient, ReplayClient,
                               ReplayMiss, request_key)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to open a network connection fails the test."""
    def guard(*args, **kwargs):
        raise RuntimeError("tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", guard)


@dataclass
class FakeLive:
    """Stands in for the real API while recording, and counts calls."""
    calls: int = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(text=f"answer to: {request.messages[-1][1]}", input_tokens=12,
                             output_tokens=3, latency_s=0.8)


def req(question="What was revenue growth?", temperature=0.0, model="m1"):
    return ModelRequest(model=model, messages=(("user", question),), temperature=temperature)


def test_same_request_same_key():
    assert request_key(req()) == request_key(req())


@pytest.mark.parametrize("change", [
    dict(question="What was cost growth?"), dict(temperature=0.7), dict(model="m2"),
])
def test_any_change_that_affects_the_answer_changes_the_key(change):
    assert request_key(req()) != request_key(req(**change))


def test_key_is_short_hex():
    key = request_key(req())
    assert len(key) == 16 and int(key, 16) >= 0


def test_record_then_replay_gives_the_same_response(tmp_path):
    live = FakeLive()
    recorded = RecordingClient(live, tmp_path).complete(req())
    replayed = ReplayClient(tmp_path).complete(req())
    assert replayed == recorded
    assert live.calls == 1                       # replay did not call the model


def test_recordings_are_readable_json(tmp_path):
    import json
    client = RecordingClient(FakeLive(), tmp_path)
    client.complete(req())
    data = json.loads((tmp_path / f"{client.recorded[0]}.json").read_text())
    assert data["request"]["model"] == "m1"
    assert data["response"]["text"].startswith("answer to")


def test_missing_recording_fails_loudly(tmp_path):
    with pytest.raises(ReplayMiss, match="m1"):
        ReplayClient(tmp_path).complete(req("never recorded"))


def test_network_is_really_blocked():
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 80), timeout=1)
