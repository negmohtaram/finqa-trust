"""Model clients: one interface, so tests can replay recorded answers offline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[tuple[str, str], ...]     # ((role, content), ...)
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


class ReplayMiss(LookupError):
    """No recording exists for this request. Record it again on purpose."""


def request_key(request: ModelRequest) -> str:
    """A stable id for a request: the same request always gives the same key, and any
    change that could change the answer (model, messages, temperature, limits) gives a new one."""
    fields = asdict(request)
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
