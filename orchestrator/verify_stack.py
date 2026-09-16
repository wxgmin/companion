"""Acceptance check for the running stack.

Exercises the real path end to end and reports ground truth rather than
inferring it: the model's resident context, actual VRAM, measured STT latency,
and a full chat round trip including spoken audio.

    orchestrator\\.venv\\Scripts\\python.exe orchestrator\\verify_stack.py

Assumes `companion.cmd` has been run. Exits non-zero if anything fails, so it
can gate a release.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import tts  # noqa: E402

TOTAL = 24576
FAILURES: list[str] = []


def ok(label: str, detail: str) -> None:
    print(f"  [PASS] {label:<26} {detail}")


def bad(label: str, detail: str) -> None:
    FAILURES.append(label)
    print(f"  [FAIL] {label:<26} {detail}")


def vram() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    used, free = (int(x.strip()) for x in out.split(","))
    return used, free


def section(name: str) -> None:
    print(f"\n{name}")
    print("-" * 68)


def main() -> int:
    print("=" * 68)
    print("  companion acceptance check")
    print("=" * 68)

    # ---------------------------------------------------------------- model
    section("language model")
    t0 = time.perf_counter()
    try:
        httpx.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.PERSONA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "think": False, "keep_alive": config.KEEP_ALIVE,
                "options": {"num_ctx": config.PERSONA_NUM_CTX, "num_predict": 5},
            },
            timeout=900,
        )
        warm_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        bad("model load", f"{type(exc).__name__}: {exc}")
        return 1

    resident = httpx.get(f"{config.OLLAMA_URL}/api/ps", timeout=20).json().get("models", [])
    if not resident:
        bad("model resident", "nothing loaded after a request")
    else:
        m = resident[0]
        ctx = m.get("context_length")
        size_gb = (m.get("size") or 0) / 1e9
        if ctx and int(ctx) >= config.PERSONA_NUM_CTX:
            ok("model resident", f"{size_gb:.1f} GB, ctx={ctx}, loaded in {warm_s:.1f}s")
        else:
            bad("model ctx", f"resident ctx={ctx}, configured {config.PERSONA_NUM_CTX}")

    # ------------------------------------------------------------- headroom
    section("vram")
    used, free = vram()
    if free >= config.MIN_FREE_VRAM_MIB:
        ok("headroom", f"{used}/{TOTAL} MiB used, {free} MiB free")
    else:
        bad("headroom", f"only {free} MiB free (want >= {config.MIN_FREE_VRAM_MIB})")

    # ------------------------------------------------------------------ tts
    section("voice")
    h = tts.healthy()
    if h.get("ok"):
        ok("chatterbox", f"ready, exaggeration={h.get('exaggeration')}")
    else:
        bad("chatterbox", str(h.get("error"))[:60])

    # ------------------------------------------------------------------ stt
    section("speech to text")
    device, compute = config.whisper_device()
    ref = config.PROJECT / "data" / "voice" / "e2e_input.wav"
    if ref.exists():
        raw = ref.read_bytes()
        try:
            t0 = time.perf_counter()
            r = httpx.post(
                "http://127.0.0.1:8090/transcribe",
                files={"file": (ref.name, raw, "audio/wav")},
                timeout=300,
            ).json()
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            r2 = httpx.post(
                "http://127.0.0.1:8090/transcribe",
                files={"file": (ref.name, raw, "audio/wav")},
                timeout=300,
            ).json()
            warm = time.perf_counter() - t0
            if warm < 1.0:
                ok("stt", f"{device}/{compute}, cold {first:.2f}s, warm {warm:.2f}s")
            else:
                bad("stt latency", f"warm {warm:.2f}s is too slow for conversation")
            print(f"         heard: {r2.get('text','')[:64]!r}")
        except Exception as exc:  # noqa: BLE001
            bad("stt", f"{type(exc).__name__}: {exc}")
    else:
        print("  [SKIP] no e2e_input.wav to transcribe")

    # ------------------------------------------------------------ roundtrip
    section("full round trip")
    try:
        t0 = time.perf_counter()
        r = httpx.post(
            "http://127.0.0.1:8090/chat",
            json={"text": "hey, just checking you're actually awake in there", "speak": True},
            timeout=600,
        ).json()
        wall = time.perf_counter() - t0
        if r.get("reply"):
            ok("reply", f"{wall:.2f}s -> {r['reply'][:52]!r}")
        else:
            bad("reply", "empty")
        if r.get("audio_wav_b64"):
            size_kb = len(base64.b64decode(r["audio_wav_b64"])) / 1024
            ok("speech audio", f"{size_kb:.1f} KB wav")
        else:
            bad("speech audio", "no audio returned")
        timing = r.get("timing", {})
        print(
            f"         llm {timing.get('llm','?')}s | tts {timing.get('tts','?')}s | "
            f"{timing.get('tok_per_sec','?')} tok/s"
        )
    except Exception as exc:  # noqa: BLE001
        bad("round trip", f"{type(exc).__name__}: {exc}")

    # ----------------------------------------------------------------- done
    print()
    print("=" * 68)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        print("=" * 68)
        return 1
    print("  ALL CHECKS PASS")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
