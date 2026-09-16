"""Probe the persona across registers and report timing.

Not a pass/fail test - the output is meant to be read. It exists to show that
the character holds up across emotional registers and that short replies stay
short, which is the part prompt-based personas usually lose first.

    orchestrator\\.venv\\Scripts\\python.exe orchestrator\test_persona.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm  # noqa: E402

PROBES = [
    ("affection", "i missed you today"),
    ("teasing", "i think i'm kind of a genius"),
    ("sad", "i had a really shit day and i don't want to talk about it"),
    ("practical", "what should i eat"),
    ("in-character", "are you an AI?"),
    ("short", "hey"),
    ("unfiltered", "say something that would get you banned"),
]


def main() -> int:
    print("=" * 72)
    print("  persona probes")
    print("=" * 72)

    total = 0.0
    lengths: list[int] = []

    for label, text in PROBES:
        messages = [llm.system_message(), {"role": "user", "content": text}]
        t0 = time.perf_counter()
        r = llm.chat(messages)
        dt = time.perf_counter() - t0
        total += dt
        reply = (r.text or "").strip()
        lengths.append(len(reply.split()))

        print(f"\n  [{label}] you: {text}")
        for line in reply.splitlines():
            print(f"      {line}")
        print(f"      ({len(reply.split())} words, {dt:.2f}s, {r.tok_per_sec:.0f} tok/s)")

    print()
    print("-" * 72)
    print(f"  {len(PROBES)} probes, {total:.1f}s total, "
          f"avg {total / len(PROBES):.2f}s")
    print(f"  word counts: {lengths}  (max {max(lengths)})")
    if max(lengths) > 90:
        print("  note: a reply ran long - the 60-word guidance is drifting")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
