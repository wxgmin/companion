"""Time one realistic persona turn end to end, with the shipped caps applied.

Uses the real prompt and the real request builder, so the numbers reflect what
a spoken turn actually costs rather than a synthetic benchmark. Streaming is
included because time-to-first-sentence is what matters to a listener - the
full reply length only decides when she stops talking.

    python orchestrator/time_turn.py [prompt]
"""

from __future__ import annotations

import sys
import time

import config
import llm

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "hey, what are you up to right now?"

print(f"caps: num_predict={config.MAX_TOKENS}  num_ctx={config.PERSONA_NUM_CTX}  "
      f"think={config.PERSONA_THINK}")
print(f"model: {config.PERSONA_MODEL}")
print()

messages = [
    llm.system_message(),
    {"role": "user", "content": PROMPT},
]

# Streaming, because that is the path the voice loop uses.
t0 = time.perf_counter()
first_piece_at = None
n_chars = 0
buf: list[str] = []

for piece in llm.chat_stream(messages):
    if first_piece_at is None:
        first_piece_at = time.perf_counter() - t0
    n_chars += len(piece)
    buf.append(piece)

total = time.perf_counter() - t0
text = "".join(buf).strip()
approx_tokens = max(1, int(n_chars / 4))

print(f"first token   : {first_piece_at:.2f}s" if first_piece_at else "no output")
print(f"full reply    : {total:.2f}s  ({n_chars} chars ~ {approx_tokens} tok, "
      f"{approx_tokens / total:.1f} tok/s)")
print()
print(f"reply ({len(text.split())} words):")
print(f"  {text[:400]}")
print()

# The listener hears the first sentence, not the whole reply.
first_sentence = text.split(".")[0].strip()
tts_guess = 3.65  # measured hot, on this machine
print("--- what the listener experiences ---")
print(f"  first word audible ~ {first_piece_at + tts_guess:.1f}s after you stop talking"
      if first_piece_at else "  n/a")
print(f"  she stops talking   ~ {total + tts_guess:.1f}s")
print()
print(f"  first sentence: {first_sentence[:90]!r}")
