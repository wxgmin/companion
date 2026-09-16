"""Serve Chatterbox as an OpenAI-compatible TTS endpoint, with HER voice baked in.

Why this exists
---------------
Open-LLM-VTuber has no native Chatterbox adapter, but it DOES ship an
`openai_tts` adapter that talks the standard OpenAI speech API:

    client.audio.speech.with_streaming_response.create(
        model=..., voice=..., input=..., response_format=..., speed=...)

which is a POST to {base_url}/audio/speech. So exposing Chatterbox behind that
shape means we can drive it from Open-LLM-VTuber with config only -- no patching.

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
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

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


def build_app(args) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load the model and encode her voice ONCE, before serving requests.
        global _engine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        print(f"reference: {args.ref}")
        _engine = Engine(Path(args.ref), args.exaggeration, args.cfg_weight, device)
        yield

    app = FastAPI(title="Chatterbox TTS (OpenAI-compatible)", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _engine is not None,
                "exaggeration": args.exaggeration}

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

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference audio (her voice)")
    ap.add_argument("--exaggeration", type=float, default=0.8)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if not Path(args.ref).exists():
        print(f"reference not found: {args.ref}")
        return 2

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
