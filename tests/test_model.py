import socket
from dataclasses import dataclass

import pytest

from finqa_trust.model import (ModelRequest, ModelResponse, RecordingClient, ReplayClient,
                               ReplayMiss, request_key)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to look up a host or open a connection fails the test, online or offline."""
    def guard(*args, **kwargs):
        raise RuntimeError("tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


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
        socket.create_connection(("example.com", 80), timeout=1)      # host lookup
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("192.0.2.1", 80), timeout=1)        # raw IP: connect


def test_groq_client_sends_the_right_request(monkeypatch):
    import io
    import json as _json
    import urllib.request
    from finqa_trust.model import GroqClient
    sent = {}

    def fake_urlopen(http, timeout, context=None):
        sent["body"] = _json.loads(http.data)
        sent["auth"] = http.headers["Authorization"]
        payload = {"choices": [{"message": {"content": "12.5"}}],
                   "usage": {"prompt_tokens": 40, "completion_tokens": 2}}
        return io.BytesIO(_json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    response = GroqClient(api_key="test-key").complete(req(temperature=0.3))
    assert sent["body"]["temperature"] == 0.3 and sent["body"]["model"] == "m1"
    assert sent["body"]["messages"] == [{"role": "user", "content": "What was revenue growth?"}]
    assert sent["auth"] == "Bearer test-key"
    assert "reasoning_effort" not in sent["body"]           # only sent when asked for
    assert (response.text, response.input_tokens, response.output_tokens) == ("12.5", 40, 2)


def test_recorded_fixture_answers_replay_and_score_offline():
    from pathlib import Path
    from finqa_trust.checker import is_correct
    from finqa_trust.data import load
    from finqa_trust.model import DEFAULT_MODEL
    from finqa_trust.prompts import direct_messages
    fixtures = Path(__file__).parent / "fixtures"
    if not (fixtures / "model").exists():
        pytest.skip("no recordings yet: run scripts/record_fixtures.py once")
    client = ReplayClient(fixtures / "model")
    records = (load("finqa", "dev", fixtures) + load("tatqa", "dev", fixtures))[:5]
    for r in records:
        response = client.complete(ModelRequest(model=DEFAULT_MODEL, messages=direct_messages(r),
                                                reasoning_effort="low"))
        assert isinstance(is_correct(response.text, r), bool)


def test_repeat_runs_get_their_own_recordings(tmp_path):
    live = FakeLive()
    recorder = RecordingClient(live, tmp_path)
    for run in range(5):
        recorder.complete(ModelRequest(model="m1", messages=(("user", "q"),), run=run))
    assert len(set(recorder.recorded)) == 5
    assert len(list(tmp_path.glob("*.json"))) == 5


def test_new_fields_at_default_keep_old_keys():
    """Recordings made before run and json_mode existed must still replay."""
    assert request_key(req()) == request_key(ModelRequest(
        model="m1", messages=(("user", "What was revenue growth?"),), run=0, json_mode=False))
    assert request_key(req()) != request_key(ModelRequest(
        model="m1", messages=(("user", "What was revenue growth?"),), run=1))


def test_resuming_client_only_pays_for_new_calls(tmp_path):
    from finqa_trust.model import ResumingClient
    live = FakeLive()
    first = ResumingClient(live, tmp_path)
    first.complete(req("q1"))
    again = ResumingClient(live, tmp_path)
    again.complete(req("q1"))
    again.complete(req("q2"))
    assert (first.live_calls, again.live_calls, live.calls) == (1, 1, 2)


def _http_error(code, payload):
    import io
    import urllib.error
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(payload.encode()))


def test_json_the_model_failed_to_write_becomes_an_unreadable_reply(monkeypatch):
    import json as _json
    import urllib.request
    from finqa_trust.model import GroqClient
    from finqa_trust.structured import parse_structured
    broken = '{"values":[{"name":"Diluted 2018","value":58,640}]'
    body = _json.dumps({"error": {"code": "json_validate_failed", "failed_generation": broken}})

    def fake_urlopen(http, timeout, context=None):
        raise _http_error(400, body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    response = GroqClient(api_key="k").complete(req())
    assert response.text == broken and response.error == "json_validate_failed"
    assert parse_structured(response.text) is None          # the labeler will call it unreadable


def test_other_client_errors_still_raise(monkeypatch):
    import urllib.request
    from finqa_trust.model import GroqClient

    def fake_urlopen(http, timeout, context=None):
        raise _http_error(400, '{"error": {"code": "model_not_found"}}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="400"):
        GroqClient(api_key="k").complete(req())


def test_a_timeout_is_retried_and_a_bad_request_is_not(monkeypatch):
    import time as _time
    from finqa_trust.model import ask
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    class Flaky:
        def __init__(self, error, failures):
            self.error, self.failures, self.calls = error, failures, 0

        def complete(self, request):
            self.calls += 1
            if self.calls <= self.failures:
                raise self.error
            return ModelResponse(text="ok")

    slow = Flaky(TimeoutError("The read operation timed out"), failures=2)
    assert ask(slow, req(), pause=0).text == "ok" and slow.calls == 3

    bad = Flaky(RuntimeError("Groq returned 400 for model 'm1': bad request"), failures=1)
    with pytest.raises(RuntimeError, match="400"):
        ask(bad, req(), pause=0)
    assert bad.calls == 1
