"""Model clients: one interface, so tests can replay recorded answers offline."""
from __future__ import annotations

import hashlib
import json
import os
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
            detail = err.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Groq returned {err.code} for model {request.model!r}: {detail}") from None
        usage = data.get("usage", {})
        return ModelResponse(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_s=round(time.perf_counter() - start, 3),
        )
