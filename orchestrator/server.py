"""HTTP + WebSocket surface for the companion.

Two ways in:

* ``POST /chat`` - text in, text plus base64 WAV out. Easy to script and test.
* ``WS /ws`` - the same loop for a face. Audio comes back as a separate event
  from the text so a lip-sync front end can start the visemes the moment the
  reply is known instead of waiting for the full waveform.

``POST /transcribe`` accepts raw audio so a browser or phone can do the STT hop
server-side instead of bundling its own model.
"""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import llm
import preflight
import router as router_mod
import tts

app = FastAPI(title="Companion", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_whisper = None


def whisper_model():
    global _whisper
    if _whisper is None:
        _whisper = config.load_whisper()
    return _whisper


def float_audio(raw: bytes, sr_in: int) -> np.ndarray:
    """Decode uploaded audio bytes to 16 kHz mono float32."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        n = int(len(mono) / sr * config.SAMPLE_RATE)
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono
        ).astype(np.float32)
    return mono


class ChatIn(BaseModel):
    text: str
    speak: bool = True


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": config.PERSONA_MODEL,
        "num_ctx": config.PERSONA_NUM_CTX,
        "voice": tts.healthy(),
        "uptime_note": "companion orchestrator",
    }


@app.get("/preflight")
def preflight_endpoint() -> dict:
    return preflight.report()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    audio = float_audio(raw, config.SAMPLE_RATE)
    t0 = time.perf_counter()
    segments, _ = whisper_model().transcribe(
        audio, beam_size=config.WHISPER_BEAM, language="en", vad_filter=True
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "seconds": round(time.perf_counter() - t0, 3)}


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    session = getattr(app.state, "router", None)
    started = time.perf_counter()
    outcome = session.respond(body.text)

    audio_b64 = None
    tts_s = 0.0
    if body.speak and outcome.reply:
        try:
            audio, tts_s = tts.synthesize(outcome.reply)
            audio_b64 = base64.b64encode(audio).decode("ascii")
        except Exception as exc:  # noqa: BLE001 - text is still useful
            print(f"  [tts] failed: {exc}", flush=True)

    return {
        "reply": outcome.reply,
        "task": outcome.task,
        "audio_wav_b64": audio_b64,
        "timing": {
            "total": round(time.perf_counter() - started, 3),
            "llm": round(outcome.seconds, 3),
            "tts": round(tts_s, 3),
            "tokens": outcome.tokens,
            "tok_per_sec": round(outcome.tok_per_sec, 1),
        },
    }


@app.post("/reset")
def reset() -> dict:
    getattr(app.state, "router").reset()
    return {"ok": True}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session = getattr(app.state, "router", None)
    await websocket.send_text(json.dumps({"type": "hello", "model": config.PERSONA_MODEL}))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type", "chat")

            if kind == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            text = (msg.get("text") or "").strip()
            if kind == "transcribe":
                audio = float_audio(base64.b64decode(msg["audio"]), config.SAMPLE_RATE)
                segments, _ = whisper_model().transcribe(
                    audio, beam_size=config.WHISPER_BEAM, language="en", vad_filter=True
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                await websocket.send_text(json.dumps({"type": "heard", "text": text}))
                if not text:
                    continue

            if not text:
                continue

            started = time.perf_counter()
            await websocket.send_text(json.dumps({"type": "thinking"}))
            outcome = session.respond(text)

            # Text first: a face can start mouth shapes before the audio lands.
            await websocket.send_text(
                json.dumps({"type": "reply", "text": outcome.reply, "task": outcome.task})
            )

            if outcome.reply and msg.get("speak", True):
                try:
                    audio, tts_s = tts.synthesize(outcome.reply)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "audio",
                                "format": "wav",
                                "audio": base64.b64encode(audio).decode("ascii"),
                                "seconds": round(tts.duration_seconds(audio), 2),
                            }
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": f"tts: {exc}"})
                    )

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "done",
                        "timing": {
                            "total": round(time.perf_counter() - started, 3),
                            "llm": round(outcome.seconds, 3),
                            "tokens": outcome.tokens,
                        },
                    }
                )
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:  # noqa: BLE001
            pass


@app.on_event("startup")
def _startup() -> None:
    app.state.router = router_mod.Router()
    print(f"companion server ready on http://{config.ORCHESTRATOR_HOST}:{config.ORCHESTRATOR_PORT}")
    print(f"  model: {config.PERSONA_MODEL}  ctx={config.PERSONA_NUM_CTX}")
    print(f"  voice: {config.TTS_URL}   (voice={config.TTS_VOICE})")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.ORCHESTRATOR_HOST,
        port=config.ORCHESTRATOR_PORT,
        log_level="warning",
    )
