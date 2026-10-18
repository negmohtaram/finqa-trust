"""Model clients: one interface, so tests can replay recorded answers offline."""
from __future__ import annotations

import hashlib
import json
import os
import random
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import certifi


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[tuple[str, str], ...]     # ((role, content), ...)
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None       # "low" / "medium" / "high" for reasoning models
    json_mode: bool = False                   # ask the API for a JSON object reply
    run: int = 0                              # repeat number when the same question is asked again on purpose; never sent


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""                           # set when the model's own output was rejected, e.g. invalid JSON


DEFAULT_MODEL = "openai/gpt-oss-120b"


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


class ReplayMiss(LookupError):
    """No recording exists for this request. Record it again on purpose."""


def request_key(request: ModelRequest) -> str:
    """A stable id for a request: the same request always gives the same key, and any
    change that could change the answer (model, messages, temperature, limits) gives a new one."""
    fields = asdict(request)
    # Fields added later are left out at their default, so older recordings keep their keys.
    for name in ("json_mode", "run"):
        if not fields[name]:
            del fields[name]
    text = json.dumps(fields, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest[:16]


@dataclass
class ReplayClient:
    """Reads recorded answers from a folder. Never touches the network."""
    folder: Path

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the recorded answer. A missing recording raises instead of calling a model."""
        path = self.folder / f"{request_key(request)}.json"
        if not path.exists():
            last = request.messages[-1][1][:60]
            raise ReplayMiss(f"no recording for {request.model}: {last!r}. Record it again.")
        data = json.loads(path.read_text())
        return ModelResponse(**data["response"])


@dataclass
class RecordingClient:
    """Wraps a real client: passes each request through and saves the answer."""
    inner: ModelClient
    folder: Path
    recorded: list[str] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Pass the request to the wrapped client and save request + response as readable JSON."""
        response = self.inner.complete(request)
        key = request_key(request)
        self.folder.mkdir(parents=True, exist_ok=True)
        record = {"request": asdict(request), "response": asdict(response)}
        (self.folder / f"{key}.json").write_text(json.dumps(record, indent=2))
        self.recorded.append(key)
        return response


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Use certifi's certificate list: some Python installs (e.g. Homebrew) ship without one.
_TLS = ssl.create_default_context(cafile=certifi.where())


def _failed_generation(code: int, body: str) -> str | None:
    if code != 400:
        return None
    try:
        error = json.loads(body).get("error", {})
    except json.JSONDecodeError:
        return None
    if error.get("code") != "json_validate_failed":
        return None
    return error.get("failed_generation", "")


@dataclass
class GroqClient:
    """Calls Groq's OpenAI-compatible chat API. The key comes from GROQ_API_KEY."""
    api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""), repr=False)
    timeout_s: float = 60.0

    def complete(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        body = {
            "model": request.model,
            "messages": [{"role": role, "content": content} for role, content in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.reasoning_effort:
            body["reasoning_effort"] = request.reasoning_effort
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        http = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "User-Agent": "finqa-trust/0.1"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(http, timeout=self.timeout_s, context=_TLS) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")
            failed = _failed_generation(err.code, detail)
            if failed is not None:
                # The model wrote broken JSON and Groq refused to return it. That is the model's
                # answer failing, not the request: keep it as a reply so it is scored, not retried.
                return ModelResponse(text=failed, error="json_validate_failed",
                                     latency_s=round(time.perf_counter() - start, 3))
            raise RuntimeError(f"Groq returned {err.code} for model {request.model!r}: {detail[:300]}") from None
        usage = data.get("usage", {})
        return ModelResponse(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_s=round(time.perf_counter() - start, 3),
        )


@dataclass
class ResumingClient:
    """Replays a request if it was already recorded, otherwise calls the real client and
    records it. An interrupted run restarts without paying again for finished calls."""
    inner: ModelClient
    folder: Path
    live_calls: int = 0

    def is_recorded(self, request: ModelRequest) -> bool:
        return (self.folder / f"{request_key(request)}.json").exists()

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            return ReplayClient(self.folder).complete(request)
        except ReplayMiss:
            self.live_calls += 1
            return RecordingClient(self.inner, self.folder).complete(request)


class DailyLimit(RuntimeError):
    """The provider's daily quota is used up. Waiting minutes will not help; rerun tomorrow."""


def ask(client: ModelClient, request: ModelRequest, pause: float = 2.0) -> ModelResponse:
    """Call with a pause after each live call. A per-minute 429 waits and retries;
    a per-day 429 stops at once, since recordings let the run resume later.
    A timeout or dropped connection is retried with capped backoff and full jitter."""
    for attempt in range(6):
        before = getattr(client, "live_calls", 0)
        try:
            response = client.complete(request)
        except (TimeoutError, ConnectionError, urllib.error.URLError) as err:
            if isinstance(err, urllib.error.HTTPError) or attempt == 5:
                raise
            time.sleep(random.uniform(0, min(60, 5 * 2 ** attempt)))
            continue
        except RuntimeError as err:
            if "429" not in str(err) or attempt == 5:
                raise
            if "per day" in str(err):
                raise DailyLimit(str(err)) from None
            time.sleep(30 * (attempt + 1))
            continue
        if getattr(client, "live_calls", 0) > before:
            time.sleep(pause)
        return response
    raise AssertionError("unreachable")
