"""Probe audio devices and verify the STT path end to end.

    python probe_audio.py            # list devices
    python probe_audio.py --mic      # record 5s from the default mic and transcribe
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

import config


def list_devices() -> int:
    print("Input devices:")
    default_in = None
    try:
        default_in = sd.query_devices(kind="input")["name"]
    except Exception as exc:  # noqa: BLE001
        print(f"  (no default input: {exc})")

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = " <- default" if dev["name"] == default_in else ""
            print(
                f"  [{idx:2}] {dev['name']}  ch={dev['max_input_channels']} "
                f"sr={int(dev['default_samplerate'])}{mark}"
            )
    return 0


def record(seconds: float) -> np.ndarray:
    print(f"Recording {seconds:.1f}s from {config.MIC_DEVICE or 'default input'} ...")
    frames = int(seconds * config.SAMPLE_RATE)
    audio = sd.rec(
        frames,
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=config.MIC_DEVICE,
        blocking=True,
    )
    sd.wait()
    return np.squeeze(audio)


def transcribe(audio: np.ndarray) -> str:
    import time

    device, compute = config.whisper_device()
    print(f"Loading whisper ({device}/{compute}) ...")
    t0 = time.perf_counter()
    model = config.load_whisper()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        beam_size=config.WHISPER_BEAM,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.perf_counter() - t0

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"  peak amplitude: {peak:.3f}")
    print(f"  transcribed in {elapsed:.2f}s ({info.duration:.1f}s audio)")
    print(f"  TEXT: {text!r}")
    if peak < 0.02:
        print("  WARNING: input looks silent - wrong device?")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="record from mic and transcribe")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--file", type=str, help="transcribe a wav file instead of the mic")
    args = ap.parse_args()

    if args.file:
        import soundfile as sf

        audio, sr = sf.read(args.file, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            duration = len(audio) / sr
            n = int(duration * config.SAMPLE_RATE)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
            ).astype(np.float32)
        transcribe(audio)
        return 0

    if args.mic:
        transcribe(record(args.seconds))
        return 0

    return list_devices()


if __name__ == "__main__":
    sys.exit(main())
