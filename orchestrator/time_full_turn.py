"""Time one complete turn the way the voice loop actually runs it.

Streams the reply, chunks it for speech, and synthesises each chunk, so the
numbers include the effects that only show up in combination: sentence-level
pipelining, the ~2.7 s per-chunk TTS fixed cost, and playback pacing.

Reports the two figures a listener actually feels:
  * time until the first word is audible
  * time until she stops talking

    python orchestrator/time_full_turn.py [prompt]
"""

from __future__ import annotations

import io
import sys
import time

import llm
import tts

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "hey, what are you up to tonight?"

messages = [llm.system_message(), {"role": "user", "content": PROMPT}]

print(f"you: {PROMPT}")
print()

# 1. Collect the reply. A real loop speaks sentence-by-sentence as it streams,
#    so this mirrors that: synthesise whenever a complete sentence has arrived.
t0 = time.perf_counter()
buf = ""
chunks: list[str] = []
audio_s = 0.0
first_audio_at = None

for piece in llm.chat_stream(messages):
    buf += piece
    ready = tts.sentences(buf)
    # Keep the trailing fragment back until the reply finishes.
    while len(ready) > 1:
        chunk = ready.pop(0)
        chunks.append(chunk)
        synth_t = time.perf_counter()
        tts.synthesize(chunk)
        if first_audio_at is None:
            first_audio_at = time.perf_counter() - t0
        audio_s += time.perf_counter() - synth_t
        buf = " ".join(ready)
        ready = tts.sentences(buf)

if buf.strip():
    chunks.append(buf.strip())
    synth_t = time.perf_counter()
    tts.synthesize(buf.strip())
    if first_audio_at is None:
        first_audio_at = time.perf_counter() - t0
    audio_s += time.perf_counter() - synth_t

total = time.perf_counter() - t0
reply = " ".join(chunks)

print(f"her: {reply}")
print()
print(f"  chunks          : {len(chunks)}")
print(f"  first word      : {first_audio_at:.2f}s  <- what you feel" if first_audio_at else "  no audio")
print(f"  done speaking   : {total:.2f}s")
print()
print("  per chunk:")
for c in chunks:
    print(f"    [{len(c):3d}] {c[:70]}")
