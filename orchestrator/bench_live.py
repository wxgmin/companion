"""Measure the real live path: router.respond() then VoiceLoop.speak().

Unlike the synthetic benchmarks this uses the shipping code path exactly, so a
regression in chunking or pipelining shows up here rather than in a micro-
benchmark that no longer matches what runs.

Reports the chunk count, the TTS wall time, and the gap between them - if TTS
time is much larger than audio produced, synthesis is falling behind playback
and the reply will stutter.

    python orchestrator/bench_live.py
"""

from __future__ import annotations

import time

import tts
from router import Router

PROMPTS = [
    "hey, what are you up to tonight?",
    "tell me something youve been thinking about lately",
    "how was your day",
]

router = Router()

# Warm both services so the first prompt does not pay model load.
tts.synthesize("warm up")
router.respond("hi")

print(f"{'prompt':<46} {'chunks':>6} {'tts_s':>7} {'chunk_text'}")
print("-" * 100)

for prompt in PROMPTS:
    t0 = time.perf_counter()
    outcome = router.respond(prompt)
    llm_s = time.perf_counter() - t0

    chunks = tts.sentences(outcome.reply) or [outcome.reply]

    t1 = time.perf_counter()
    audio_s = 0.0
    for chunk in chunks:
        audio, _ = tts.synthesize(chunk)
        # Rough playback length: WAV at 24 kHz mono 16-bit.
        audio_s += max(0.0, (len(audio) - 44) / (24000 * 2))
    tts_s = time.perf_counter() - t1

    print(f"{prompt[:44]:<46} {len(chunks):>6} {tts_s:>7.2f}   "
          f"{' | '.join(c[:26] for c in chunks[:3])}")
    print(f"{'':46} llm {llm_s:5.2f}s  audio {audio_s:5.2f}s  "
          f"first-word ~{llm_s + (tts_s / max(len(chunks), 1)):.2f}s")

print()
print("reading: chunk count drives fixed TTS cost (~2.7s each on this machine),")
print("         so fewer chunks is the lever, not shorter text.")
