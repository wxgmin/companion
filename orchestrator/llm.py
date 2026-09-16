"""Ollama client for the persona brain.

Handles the two things that are easy to get wrong with this model:

* ``think`` must be passed explicitly. Qwen3.8 defaults to a reasoning trace
  that streams into a separate ``thinking`` field and leaves ``content`` empty
  until it finishes, which roughly triples time-to-first-token.
* ``num_ctx`` must be sent on every request. Ollama defaults it far lower than
  what we measured as safe, and a context that overflows VRAM spills to the CPU
  and quietly degrades tool-call reliability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

import config


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    tokens: int = 0
    seconds: float = 0.0

    @property
    def tok_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


def _payload(messages: list[dict[str, Any]], stream: bool, tools=None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.PERSONA_MODEL,
        "messages": messages,
        "stream": stream,
        "think": config.PERSONA_THINK,
        "keep_alive": config.KEEP_ALIVE,
        "options": {
            "num_ctx": config.PERSONA_NUM_CTX,
            "num_predict": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
        },
    }
    if tools:
        body["tools"] = tools
    return body


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 300.0,
) -> Reply:
    """One non-streaming turn. Returns text and/or tool calls."""
    import time

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{config.OLLAMA_URL}/api/chat",
            json=_payload(messages, stream=False, tools=tools),
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.perf_counter() - t0

    msg = data.get("message", {})
    calls = [
        ToolCall(
            name=c["function"]["name"],
            arguments=c["function"].get("arguments") or {},
        )
        for c in (msg.get("tool_calls") or [])
    ]
    return Reply(
        text=(msg.get("content") or "").strip(),
        tool_calls=calls,
        thinking=msg.get("thinking") or "",
        tokens=int(data.get("eval_count") or 0),
        seconds=float(data.get("eval_duration") or 0) / 1e9 or elapsed,
    )


def chat_stream(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 300.0,
) -> Iterator[str]:
    """Yield content deltas as they arrive, for sentence-level TTS pipelining."""
    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{config.OLLAMA_URL}/api/chat",
            json=_payload(messages, stream=True, tools=tools),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (chunk.get("message") or {}).get("content")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break


def system_message(extra: str = "") -> dict[str, str]:
    content = config.PERSONA_PROMPT
    if extra:
        content = f"{content}\n\n{extra}"
    return {"role": "system", "content": content}


def unload(keep_alive: int = 0) -> None:
    """Release the model's VRAM. Used by preflight before loading anything else."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={"model": config.PERSONA_MODEL, "messages": [], "keep_alive": keep_alive},
            )
    except Exception:  # noqa: BLE001 - best effort
        pass


def resident() -> list[dict[str, Any]]:
    """Whatever Ollama currently has loaded, with its context size."""
    try:
        with httpx.Client(timeout=15.0) as client:
            return client.get(f"{config.OLLAMA_URL}/api/ps").json().get("models", [])
    except Exception:  # noqa: BLE001
        return []
