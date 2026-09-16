"""Full-duplex voice loop: mic -> VAD -> whisper -> persona -> Chatterbox -> speakers.

Turn taking is driven entirely by voice activity, so there is no push-to-talk:
speech starts an utterance, roughly three quarters of a second of silence ends
it, and speaking while she is talking cuts her off. All four stages are timed
and printed so latency regressions are visible instead of guessed at.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import webrtcvad

import config
import llm
import tts
from router import Router


@dataclass
class Turn:
    heard: str
    said: str
    stt_s: float
    llm_s: float
    tts_s: float
    tokens: int
    task: str | None = None

    @property
    def total_s(self) -> float:
        return self.stt_s + self.llm_s + self.tts_s


class VoiceLoop:
    def __init__(self, router: Router | None = None) -> None:
        self.router = router or Router()
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.audio_q: queue.Queue[bytes] = queue.Queue()
        self.turns: list[Turn] = []
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._whisper = None

    # ------------------------------------------------------------- whisper
    @property
    def whisper(self):
        if self._whisper is None:
            self._whisper = config.load_whisper()
        return self._whisper

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.whisper.transcribe(
            audio,
            beam_size=config.WHISPER_BEAM,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # ------------------------------------------------------------ capture
    def _mic_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            pass
        self.audio_q.put(bytes(indata))

    def _frame_iter(self):
        """Yield 30 ms frames of int16 audio, forever."""
        blocksize = int(config.SAMPLE_RATE * config.VAD_FRAME_MS / 1000)
        with sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=blocksize,
            device=config.MIC_DEVICE,
            dtype="int16",
            channels=1,
            callback=self._mic_callback,
        ):
            while not self._stop.is_set():
                try:
                    yield self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

    def listen_for_utterance(self) -> np.ndarray | None:
        """Block until one spoken utterance is captured, then return it as float32."""
        ring: collections.deque[bytes] = collections.deque(maxlen=config.VAD_START_FRAMES)
        voiced: list[bytes] = []
        started = False
        silence_run = 0
        frames_total = 0
        max_frames = int(config.VAD_MAX_UTTERANCE_S * 1000 / config.VAD_FRAME_MS)
        min_frames = int(config.VAD_MIN_UTTERANCE_S * 1000 / config.VAD_FRAME_MS)

        for frame in self._frame_iter():
            if self._stop.is_set():
                return None

            is_speech = False
            try:
                is_speech = self.vad.is_speech(frame, config.SAMPLE_RATE)
            except Exception:  # noqa: BLE001 - malformed frame length
                continue

            # Barge-in: any speech while she talks stops playback.
            if is_speech and self._speaking.is_set():
                sd.stop()

            if not started:
                ring.append(frame)
                if is_speech:
                    silence_run = 0
                    voice_so_far = sum(
                        1
                        for f in ring
                        if self._safe_is_speech(f)
                    )
                    if voice_so_far >= config.VAD_START_FRAMES:
                        started = True
                        voiced = list(ring)
                        ring.clear()
                        print("  [listening]", flush=True)
                continue

            voiced.append(frame)
            frames_total += 1
            silence_run = 0 if is_speech else silence_run + 1

            if silence_run >= config.VAD_SILENCE_FRAMES or frames_total >= max_frames:
                if len(voiced) < min_frames:
                    started, voiced, frames_total, silence_run = False, [], 0, 0
                    continue
                raw = b"".join(voiced)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                return audio
        return None

    def _safe_is_speech(self, frame: bytes) -> bool:
        try:
            return self.vad.is_speech(frame, config.SAMPLE_RATE)
        except Exception:  # noqa: BLE001
            return False

    # -------------------------------------------------------------- output
    def speak(self, text: str) -> tuple[float, float]:
        """Speak a reply, generating the next sentence while the current plays.

        Blocking here would leave a silent gap of roughly the reply's length,
        since Chatterbox runs slower than real time. Instead a producer thread
        keeps one sentence ahead of the speakers.
        """
        import io
        import queue as _queue
        import soundfile as sf

        chunks = tts.sentences(text) or [text]
        ready: _queue.Queue = _queue.Queue()
        first_audio_s = None

        def produce() -> None:
            for chunk in chunks:
                try:
                    audio, _ = tts.synthesize(chunk)
                    ready.put((chunk, audio))
                except Exception as exc:  # noqa: BLE001
                    print(f"  [tts] chunk failed ({exc}); skipping", flush=True)
            ready.put(None)

        worker = threading.Thread(target=produce, daemon=True)
        t_start = time.perf_counter()
        worker.start()

        total_tts = 0.0
        played = 0.0
        self._speaking.set()
        try:
            while True:
                item = ready.get()
                if item is None:
                    break
                _, audio = item
                if first_audio_s is None:
                    first_audio_s = time.perf_counter() - t_start
                data, sr = sf.read(io.BytesIO(audio), dtype="float32")
                played += len(data) / sr
                sd.play(data, sr)
                sd.wait()
        finally:
            self._speaking.clear()
            worker.join(timeout=5)

        total_tts = time.perf_counter() - t_start
        if first_audio_s is not None:
            print(
                f"  [voice] first audio after {first_audio_s:.2f}s, "
                f"{played:.1f}s spoken (chunks={len(chunks)})",
                flush=True,
            )
        return total_tts, played

    # ----------------------------------------------------------------- run
    def handle(self, text: str) -> Turn:
        t0 = time.perf_counter()
        stt_s = 0.0  # already transcribed by the caller

        outcome = self.router.respond(text)
        llm_s = outcome.seconds
        print(f"  [her] {outcome.reply}", flush=True)

        tts_s, played = self.speak(outcome.reply)
        turn = Turn(
            heard=text,
            said=outcome.reply,
            stt_s=stt_s,
            llm_s=llm_s,
            tts_s=tts_s,
            tokens=outcome.tokens,
            task=outcome.task,
        )
        print(
            f"  [timing] llm {llm_s:.2f}s | tts {tts_s:.2f}s ({played:.1f}s audio) | "
            f"total {time.perf_counter() - t0:.2f}s | "
            f"{outcome.tokens} tok @ {outcome.tok_per_sec:.1f} tok/s",
            flush=True,
        )
        self.turns.append(turn)
        return turn

    def run(self) -> None:
        print("voice loop up - start talking (ctrl-c to stop)\n", flush=True)
        try:
            while not self._stop.is_set():
                audio = self.listen_for_utterance()
                if audio is None:
                    continue
                t0 = time.perf_counter()
                text = self.transcribe(audio)
                stt_s = time.perf_counter() - t0
                if not text:
                    print("  [heard nothing intelligible]", flush=True)
                    continue
                print(f"  [you] {text}  (stt {stt_s:.2f}s)", flush=True)
                turn = self.handle(text)
                turn.stt_s = stt_s
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
            print("\nsession ended", flush=True)

    def close(self) -> None:
        self._stop.set()


def main() -> int:
    import preflight

    if preflight.main() != 0:
        print("\npreflight failed - fix the above before starting the loop")
        return 1
    print()
    VoiceLoop().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
