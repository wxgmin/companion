"""Measure free VRAM with the whole stack resident at a given context size.

Run this before changing PERSONA_NUM_CTX. The binding constraint on this
machine is not whether a context fits, but whether it leaves enough headroom
that nothing spills to the CPU under load.
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import config  # noqa: E402


def used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    return int(out.split()[0])


def load_model(ctx: int) -> dict:
    httpx.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.PERSONA_MODEL,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "think": False,
            "keep_alive": "2m",
            "options": {"num_ctx": ctx, "num_predict": 4},
        },
        timeout=600,
    )
    ps = httpx.get(f"{config.OLLAMA_URL}/api/ps", timeout=20).json().get("models", [])
    return next((m for m in ps if m.get("name") == config.PERSONA_MODEL), {})


def main() -> int:
    baseline = used()
    print(f"baseline (desktop + chatterbox + whisper): {baseline} MiB")

    for ctx in (32768, 49152, 65536):
        info = load_model(ctx)
        time.sleep(4)
        now = used()
        free = 24576 - now
        size_gb = (info.get("size") or 0) / 1e9
        trained = info.get("context_length")
        verdict = "COMFORTABLE" if free >= 1500 else ("OK" if free >= 900 else "TIGHT")
        print(
            f"  num_ctx={ctx:<7} model={size_gb:4.1f}GB  total={now:6} MiB  "
            f"free={free:5} MiB   {verdict}"
        )
        if trained and int(trained) < ctx:
            print(f"      note: model was trained for {trained}")

    httpx.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={"model": config.PERSONA_MODEL, "messages": [], "keep_alive": 0},
        timeout=60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
