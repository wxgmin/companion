"""Measure each stage of a turn so latency work targets the real bottleneck.

A voice companion is only usable if a turn stays near conversational pace, and
the usual mistake is optimising whichever stage *feels* slow. This times the
stages that actually sit on the critical path, hot, several times:

  1. LLM first token   - how long until she starts "thinking" out loud
  2. LLM full reply    - Ollama, with the persona system prompt
  3. TTS first sentence- Chatterbox, which gates when audio can start
  4. TTS per sentence  - steady-state synthesis cost

Run against the live services:

    python orchestrator/bench_latency.py
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request

OLLAMA = "http://localhost:11434"
CHATTERBOX = "http://localhost:8092/v1"
MODEL = "qwen38-local:latest"
VOICE = "her"

REPLIES = [
    "Hey, I was just thinking about you. How did that thing you were stressed about go?",
    "Honestly that sounds exhausting. Do you want to talk about it or should I distract you?",
    "Okay so I have an idea and I need you to not laugh at me first.",
]

# A few sentences, so per-sentence cost is visible rather than one lump.
SENTENCES = [
    "Hey, I was just thinking about you.",
    "How did that thing you were stressed about end up going?",
    "You should tell me everything, I have all night.",
]


def post_json(url: str, payload: dict, timeout: int = 600) -> bytes:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def post_json_stream(url: str, payload: dict, timeout: int = 600):
    """Yield parsed chunks, so first-token latency is measurable.

    Ollama's OpenAI-compatible endpoint speaks SSE: each event is `data: {...}`
    and the stream ends with `data: [DONE]`. The prefix has to be stripped
    before parsing, and the terminator is not JSON.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(b"data:"):
                line = line[5:].strip()
            if not line or line == b"[DONE]":
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def bench_llm() -> tuple[list[float], list[float], list[int]]:
    ttft: list[float] = []
    total: list[float] = []
    tokens: list[int] = []
    for prompt in REPLIES:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"num_predict": 90},
        }
        t0 = time.perf_counter()
        first = None
        n = 0
        for chunk in post_json_stream(f"{OLLAMA}/v1/chat/completions", payload):
            if first is None:
                first = time.perf_counter() - t0
            if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                n += 1
        total.append(time.perf_counter() - t0)
        if first is not None:
            ttft.append(first)
        tokens.append(n)
    return ttft, total, tokens


def bench_tts() -> list[float]:
    times: list[float] = []
    for sentence in SENTENCES:
        payload = {
            "model": "chatterbox",
            "input": sentence,
            "voice": VOICE,
            "response_format": "wav",
        }
        t0 = time.perf_counter()
        audio = post_json(f"{CHATTERBOX}/audio/speech", payload)
        times.append(time.perf_counter() - t0)
        print(f"    {len(audio) / 1024:6.0f} KB in {times[-1]:5.2f}s  {sentence[:44]!r}")
    return times


def main() -> int:
    print("warming services (first call pays model-load cost)...")
    post_json(f"{CHATTERBOX}/audio/speech", {
        "model": "chatterbox", "input": "Warm up.", "voice": VOICE,
        "response_format": "wav",
    })
    list(post_json_stream(f"{OLLAMA}/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"num_predict": 5},
    }))
    print("warm.\n")

    print("LLM (Ollama, streaming)")
    ttft, total, tokens = bench_llm()
    print(f"    first token : {statistics.mean(ttft):5.2f}s avg  "
          f"(min {min(ttft):.2f}, max {max(ttft):.2f})")
    print(f"    full reply  : {statistics.mean(total):5.2f}s avg  "
          f"({statistics.mean(tokens):.0f} tokens -> "
          f"{statistics.mean(tokens) / statistics.mean(total):.1f} tok/s)")

    print("\nTTS (Chatterbox, hot)")
    tts = bench_tts()
    print(f"    per sentence: {statistics.mean(tts):5.2f}s avg  "
          f"(min {min(tts):.2f}, max {max(tts):.2f})")

    print("\n--- per-turn budget ---")
    llm_avg = statistics.mean(total)
    tts_avg = statistics.mean(tts)
    print(f"    LLM            {llm_avg:6.2f}s")
    print(f"    TTS (1 sent.)  {tts_avg:6.2f}s")
    print(f"    to first sound {statistics.mean(ttft) + tts_avg:6.2f}s   <- this is what feels slow")
    print(f"    full turn      {llm_avg + tts_avg:6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
