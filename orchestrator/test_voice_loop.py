"""Verify the voice loop's turn-taking without needing a real microphone.

Feeds a known speech clip into the loop in real 30 ms frames, exactly as the
sound-card callback would, and asserts that the VAD finds the start and end of
the utterance. This exercises the actual listen_for_utterance path rather than
mocking it, so a regression in the frame accounting shows up here.

    orchestrator\\.venv\\Scripts\\python.exe orchestrator\\test_voice_loop.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import voice_loop  # noqa: E402

CLIP = config.PROJECT / "data" / "voice" / "e2e_input.wav"
FRAME = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<32} {detail}")
    if not ok:
        FAILURES.append(label)


def make_frames(path: Path) -> list[bytes]:
    """Load a clip and emit it as 30 ms int16 frames, with silence padding."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        n = int(len(audio) / sr * config.SAMPLE_RATE)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    # Real capture always has room tone either side of speech.
    lead = [np.zeros(FRAME, dtype=np.int16)] * 8
    tail = [np.zeros(FRAME, dtype=np.int16)] * 40
    body = [pcm[i : i + FRAME] for i in range(0, len(pcm) - FRAME, FRAME)]
    frames = lead + body + tail
    return [f.tobytes() for f in frames]


def main() -> int:
    print("=" * 68)
    print("  voice loop turn-taking test")
    print("=" * 68)

    if not CLIP.exists():
        print(f"  no clip at {CLIP} - nothing to feed")
        return 1

    vl = voice_loop.VoiceLoop.__new__(voice_loop.VoiceLoop)
    vl.vad = voice_loop.webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    vl.audio_q = queue.Queue()
    vl._stop = threading.Event()
    vl._speaking = threading.Event()
    vl._whisper = None
    vl.turns = []
    vl.router = None

    frames = make_frames(CLIP)
    print(f"  clip: {CLIP.name}, {len(frames)} frames of {config.VAD_FRAME_MS}ms")

    # Stand in for the sound-card callback, and replace the frame source so the
    # real microphone is never opened. Without this the loop also consumes live
    # room audio and the captured length is meaningless.
    def fake_frames():
        while not vl._stop.is_set():
            try:
                yield vl.audio_q.get(timeout=0.5)
            except queue.Empty:
                return

    vl._frame_iter = fake_frames  # type: ignore[method-assign]

    def feed() -> None:
        for f in frames:
            vl.audio_q.put(f)
            time.sleep(config.VAD_FRAME_MS / 1000.0)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()

    t0 = time.perf_counter()
    audio = vl.listen_for_utterance()
    elapsed = time.perf_counter() - t0

    check("utterance detected", audio is not None, f"in {elapsed:.1f}s")
    if audio is None:
        print("\n  VAD never saw an utterance boundary")
        return 1

    dur = len(audio) / config.SAMPLE_RATE
    check(
        "captured length sane",
        config.VAD_MIN_UTTERANCE_S <= dur <= config.VAD_MAX_UTTERANCE_S,
        f"{dur:.2f}s captured",
    )
    check(
        "audio is not silent",
        float(np.max(np.abs(audio))) > 0.02,
        f"peak {float(np.max(np.abs(audio))):.3f}",
    )

    # The captured audio should still transcribe to something recognisable.
    device, compute = config.whisper_device()
    print(f"\n  transcribing with {device}/{compute} ...")
    t0 = time.perf_counter()
    text = vl.transcribe(audio)
    stt_s = time.perf_counter() - t0
    check("transcribed", bool(text), f"{stt_s:.2f}s")
    print(f"         heard: {text[:68]!r}")

    print()
    print("=" * 68)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  TURN-TAKING OK")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
