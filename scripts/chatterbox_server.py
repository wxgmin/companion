"""Serve Chatterbox as an OpenAI-compatible voice endpoint, with HER voice baked in.

Why this exists
---------------
Open-LLM-VTuber has no native Chatterbox adapter, but it DOES ship an
`openai_tts` adapter that talks the standard OpenAI speech API:

    client.audio.speech.with_streaming_response.create(
        model=..., voice=..., input=..., response_format=..., speed=...)

which is a POST to {base_url}/audio/speech. So exposing Chatterbox behind that
shape means we can drive it from Open-LLM-VTuber with config only -- no patching.

The same reasoning covers speech-to-text: AIRI's `openai-audio` provider takes a
single base URL and builds both a speech and a transcription client from it, so
transcription has to live here too rather than on a second port. Both directions
share one endpoint and one model store.

Performance note
----------------
Chatterbox's `generate(audio_prompt_path=...)` re-encodes the reference audio on
EVERY call. Instead we call `prepare_conditionals()` once at startup and then
`generate()` without a prompt path, which reuses the cached speaker embedding
and speech tokens. That is the difference between usable and not.

Usage:
    python chatterbox_server.py --ref REF.wav [--exaggeration 0.8] [--port 8092]
"""

import argparse
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

# Resolved once at import. Browsers upload webm/opus, which libsndfile cannot
# read inside a webm container, so ffmpeg is the only decoder that reliably
# handles what the microphone actually produces.
_FFMPEG = shutil.which("ffmpeg")

# ---------------------------------------------------------------------------
# request/response models (OpenAI speech API shape)
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    # non-standard extras, ignored by OLV but handy for manual testing
    exaggeration: float | None = None
    cfg_weight: float | None = None


# ---------------------------------------------------------------------------
# speech-to-text
# ---------------------------------------------------------------------------

_whisper = None
_whisper_lock = threading.Lock()


def _resolve_whisper(path: Path) -> Path:
    """Return a directory that actually contains model.bin.

    The model is stored in HuggingFace cache layout, so the weights live under
    snapshots/<rev>/. faster-whisper needs the flat directory and taking the
    cache root fails with "Unable to open file 'model.bin'".
    """
    if (path / "model.bin").exists():
        return path
    hits = sorted(path.glob("models--*/snapshots/*/model.bin"))
    return hits[-1].parent if hits else path


def load_whisper(model_dir):
    """Load faster-whisper lazily so a missing install cannot break TTS.

    Speech output is the part the companion cannot function without, so ASR
    failing to import must degrade to a 503 on one endpoint rather than taking
    the whole server down. Callers pass False to disable ASR entirely.
    """
    global _whisper
    if model_dir is False:
        return None
    if _whisper is not None:
        return _whisper
    with _whisper_lock:
        if _whisper is not None:
            return _whisper
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return None

        default_dir = Path(r"C:\Users\Waiz\LocalLLMs\Voice Models") / \
            "ASR - Speech to Text" / "faster-whisper-large-v3-turbo"
        root = _resolve_whisper(Path(model_dir) if model_dir else default_dir)

        # int8_float16 keeps this near 1.2 GB, which matters because the 27B and
        # Chatterbox are already resident on the same 24 GB card.
        for device, compute in (("cuda", "int8_float16"), ("cpu", "int8")):
            try:
                _whisper = WhisperModel(str(root), device=device, compute_type=compute)
                print(f"  whisper ready: device={device} compute={compute}")
                return _whisper
            except Exception as exc:  # noqa: BLE001
                print(f"  whisper {device} unavailable ({type(exc).__name__})")
        return None


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

_engine = None
_lock = threading.Lock()


class Engine:
    def __init__(self, ref: Path, exaggeration: float, cfg_weight: float, device: str):
        from chatterbox.tts import ChatterboxTTS

        self.ref = ref
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

        t0 = time.time()
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sr = self.model.sr

        # encode her voice ONCE
        self.model.prepare_conditionals(str(ref), exaggeration=exaggeration)
        print(f"  model + voice prompt ready in {time.time() - t0:.1f}s "
              f"(sr={self.sr}, exaggeration={exaggeration})")

    def synth(self, text: str, exaggeration: float | None, cfg_weight: float | None,
              speed: float) -> bytes:
        ex = self.exaggeration if exaggeration is None else exaggeration
        cfg = self.cfg_weight if cfg_weight is None else cfg_weight
        with _lock:
            wav = self.model.generate(text, exaggeration=ex, cfg_weight=cfg)

        audio = wav.squeeze().detach().cpu().float().numpy()

        # Chatterbox has no speed control; resample to change duration.
        if abs(speed - 1.0) > 1e-3:
            import librosa

            audio = librosa.resample(audio, orig_sr=self.sr,
                                     target_sr=int(self.sr / speed))

        buf = io.BytesIO()
        sf.write(buf, audio, self.sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def _decode_to_16k(raw: bytes, filename: str | None) -> np.ndarray:
    """Decode any uploaded audio to 16 kHz mono float32.

    faster-whisper assumes 16 kHz when it is handed a bare array - it takes no
    sample-rate argument - so resampling here is not optional. Feeding it 24 or
    48 kHz audio does not raise: the model hears sped-up speech and returns
    confident nonsense, which is far harder to spot than an outright error.

    ffmpeg is preferred because it handles every container a browser might
    produce and resamples in the same pass. soundfile and librosa remain as
    fallbacks for the plain-WAV case where ffmpeg is unavailable.
    """
    suffix = Path(filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        if _FFMPEG:
            # -ar 16000 -ac 1 emits exactly what whisper wants, so no second
            # resampling step is needed or wanted.
            proc = subprocess.run(
                [
                    _FFMPEG, "-nostdin", "-loglevel", "error",
                    "-i", tmp_path,
                    "-f", "f32le", "-ac", "1", "-ar", "16000",
                    "-",
                ],
                capture_output=True,
            )
            if proc.returncode == 0 and proc.stdout:
                audio = np.frombuffer(proc.stdout, dtype=np.float32)
                if audio.size:
                    print(
                        f"  transcript: {audio.size / 16000:.1f}s via ffmpeg "
                        f"({suffix}, {len(raw) // 1024} KB)",
                        flush=True,
                    )
                    return audio
            print(f"  ffmpeg failed rc={proc.returncode}; falling back", flush=True)

        data, sr = sf.read(tmp_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa

            data = librosa.resample(
                np.asarray(data, dtype="float32"), orig_sr=sr, target_sr=16000
            )
        print(
            f"  transcript: {len(data) / 16000:.1f}s via soundfile ({sr}Hz src)",
            flush=True,
        )
        return np.asarray(data, dtype="float32")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_app(args) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load the model and encode her voice ONCE, before serving requests.
        global _engine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        print(f"reference: {args.ref}")
        print(f"ffmpeg: {_FFMPEG or 'NOT FOUND (webm/opus uploads will fail)'}")
        _engine = Engine(Path(args.ref), args.exaggeration, args.cfg_weight, device)
        yield

    app = FastAPI(title="Chatterbox TTS (OpenAI-compatible)", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _engine is not None,
                "exaggeration": args.exaggeration,
                "transcription": args.whisper_dir is not False}

    @app.get("/v1/audio/voices")
    def voices():
        return {"voices": ["her"], "uploaded_voices": []}

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        if _engine is None:
            raise HTTPException(status_code=503, detail="model still loading")
        text = (req.input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="input is empty")
        try:
            audio = _engine.synth(text, req.exaggeration, req.cfg_weight,
                                  req.speed or 1.0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}")
        return Response(content=audio, media_type="audio/wav")

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ):
        """OpenAI-shaped transcription. AIRI's hearing provider posts here."""
        asr = load_whisper(args.whisper_dir)
        if asr is None:
            raise HTTPException(
                status_code=503,
                detail="transcription unavailable (faster-whisper not installed)",
            )

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty audio")

        data = _decode_to_16k(raw, file.filename)
        if data.size == 0:
            raise HTTPException(status_code=400, detail="decoded to zero samples")

        # vad_filter drops non-speech. Without it whisper invents fluent text
        # from silence and background noise, which reads as "worked" while
        # returning something the user never said.
        try:
            segments, _info = asr.transcribe(
                data,
                language=language or None,
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"transcription failed: {exc}")

        # An empty string here is the most likely cause of AIRI's "failed to
        # transcribe" error: the HTTP call still returns 200, so the failure is
        # invisible from the transport side and only shows up as blank text.
        # Logging the exact value makes that distinguishable from a bad upload.
        print(
            f"  -> text={text!r} "
            f"(file={file.filename!r} bytes={len(raw)} "
            f"rms={float(np.sqrt(np.mean(data ** 2))):.5f})",
            flush=True,
        )

        if response_format == "text":
            return Response(content=text, media_type="text/plain")
        return {"text": text}

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference audio (her voice)")
    ap.add_argument("--exaggeration", type=float, default=0.8)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--whisper-dir",
        default=None,
        help="faster-whisper model dir; defaults to the local model store",
    )
    ap.add_argument(
        "--no-transcription",
        action="store_true",
        help="serve speech only (skips loading ASR, saving ~1.2 GB VRAM)",
    )
    args = ap.parse_args()
    if args.no_transcription:
        args.whisper_dir = False

    if not Path(args.ref).exists():
        print(f"reference not found: {args.ref}")
        return 2

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
