"""Staged VRAM accounting: measure each component as it is added.

Run after stopping the server so nothing is double-counted.

    orchestrator\\.venv\\Scripts\\python.exe orchestrator\\measure_stack.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

TOTAL = 24576


def used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    return int(out.split()[0])


def row(label: str, base: int) -> int:
    now = used()
    print(f"  {label:<38} {now:6} MiB   (+{now - base:5})   free {TOTAL - now:5}")
    return now


def unload_model() -> None:
    try:
        httpx.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={"model": config.PERSONA_MODEL, "messages": [], "keep_alive": 0},
            timeout=60,
        )
        time.sleep(5)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    print("staged VRAM measurement")
    print("=" * 72)
    unload_model()

    base = row("A. everything released", 0)

    # --- whisper on the GPU ------------------------------------------------
    import config as _c

    for dev, comp, label in (
        ("cuda", "float16", "B1. + whisper cuda/float16"),
        ("cuda", "int8_float16", "B2. + whisper cuda/int8_float16"),
    ):
        try:
            from faster_whisper import WhisperModel

            m = WhisperModel(_c.WHISPER_MODEL, device=dev, compute_type=comp)
            time.sleep(4)
            row(label, base)
            del m
            time.sleep(4)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<38} FAILED: {type(exc).__name__}")
            time.sleep(2)

    print()
    print("  LLM at each context (whisper NOT resident):")
    for ctx in (16384, 32768, 65536, 98304, 131072):
        unload_model()
        try:
            httpx.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={
                    "model": config.PERSONA_MODEL,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False, "think": False, "keep_alive": "2m",
                    "options": {"num_ctx": ctx, "num_predict": 4},
                },
                timeout=900,
            )
            time.sleep(4)
            now = used()
            free = TOTAL - now
            print(
                f"    num_ctx={ctx:<7} total={now:6} MiB   free={free:5} MiB   "
                f"{'COMFORTABLE' if free >= 1800 else ('OK' if free >= 1200 else 'TIGHT')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    num_ctx={ctx:<7} FAILED: {type(exc).__name__}: {str(exc)[:50]}")

    print()
    print("  note: subtract ~1300 MiB from 'free' above if whisper runs on GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
