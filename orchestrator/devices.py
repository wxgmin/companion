"""Show which microphone and speaker the companion will actually use.

A wrong default device is the most common reason a working stack appears dead:
the loop opens something silent and reports "heard nothing intelligible"
forever. Worth checking before blaming the pipeline.

    python orchestrator/devices.py
"""

from __future__ import annotations

import sounddevice as sd

print("input devices")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        mark = "  <- default" if i == sd.default.device[0] else ""
        print(f"  [{i:2}] {d['name'][:54]}{mark}")

print()
print("output devices")
for i, d in enumerate(sd.query_devices()):
    if d["max_output_channels"] > 0:
        mark = "  <- default" if i == sd.default.device[1] else ""
        print(f"  [{i:2}] {d['name'][:54]}{mark}")

print()
try:
    print(f"default input : {sd.query_devices(kind='input')['name']}")
    print(f"default output: {sd.query_devices(kind='output')['name']}")
except Exception as exc:  # noqa: BLE001
    print(f"could not resolve defaults: {exc}")

# A short capture proves the device can actually open and deliver frames,
# which is what the loop needs - not just that it is listed.
print()
try:
    import numpy as np

    rec = sd.rec(int(1.0 * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    peak = float(np.abs(rec).max())
    print(f"1s test capture: peak {peak:.4f}  "
          f"({'silent - check the mic' if peak < 1e-4 else 'signal present'})")
except Exception as exc:  # noqa: BLE001
    print(f"test capture failed: {exc}")
