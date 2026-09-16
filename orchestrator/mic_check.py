"""Confirm the live microphone is actually delivering usable frames.

The voice loop can sit silent for two very different reasons: nobody spoke, or
the input stream opened but delivers nothing. This counts frames and how many
the VAD calls speech, so the difference is visible.

    python orchestrator/mic_check.py [seconds]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd
import webrtcvad

import config

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0

print(f"config.MIC_DEVICE = {config.MIC_DEVICE!r}")
print(f"config.SAMPLE_RATE = {config.SAMPLE_RATE}")
print(f"VAD aggressiveness = {config.VAD_AGGRESSIVENESS}")
print(f"resolved input     = {sd.query_devices(config.MIC_DEVICE, kind='input')['name']}")
print()

vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
block = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
frames = 0
voiced = 0
peak = 0.0
t0 = time.perf_counter()

with sd.RawInputStream(
    samplerate=config.SAMPLE_RATE,
    blocksize=block,
    device=config.MIC_DEVICE,
    dtype="int16",
    channels=1,
) as stream:
    while time.perf_counter() - t0 < SECONDS:
        data, overflowed = stream.read(block)
        frames += 1
        raw = bytes(data)
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        peak = max(peak, float(np.abs(a).max()))
        try:
            if vad.is_speech(raw, config.SAMPLE_RATE):
                voiced += 1
        except Exception:  # noqa: BLE001
            pass

elapsed = time.perf_counter() - t0
print(f"{frames} frames in {elapsed:.1f}s  "
      f"(expected ~{SECONDS * 1000 / config.VAD_FRAME_MS:.0f})")
print(f"peak amplitude : {peak:.4f}")
print(f"VAD speech     : {voiced}/{frames} frames ({voiced / max(frames, 1) * 100:.0f}%)")
print()
print(
    "VERDICT: mic is delivering audio"
    if frames > 0
    else "VERDICT: NO FRAMES - the input stream is not delivering"
)
