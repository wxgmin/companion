"""Chatterbox TTS client.

Chatterbox is served as an OpenAI-compatible endpoint on :8092, so this is a
thin wrapper that also normalises text the way a chat message should be spoken
(emoji and markdown stripped, since they are written for the eye, not the ear).
"""

from __future__ import annotations

import io
import re
import time

import httpx

import config

_EMOJI = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff" "\U00002b00-\U00002bff" "\ufe0f" "\u2764" "]+",
    flags=re.UNICODE,
)
_MD = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]*)\*\*"), r"\1"),
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"\1"),
    (re.compile(r"^#{1,6}\s*", re.M), ""),
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),
    (re.compile(r"[ \t]{2,}"), " "),
]


def spellable(text: str) -> str:
    """Strip what a TTS voice should not read aloud, without changing meaning."""
    out = text
    for pattern, repl in _MD:
        out = pattern.sub(repl, out)
    out = _EMOJI.sub("", out)
    out = out.replace("<3", " love ").replace("&", " and ")
    # Reduce stretched words ("sooo" -> "soo") so the vocoder does not wobble.
    out = re.sub(r"(\w)\1{2,}", r"\1\1", out)
    return re.sub(r"\s+", " ", out).strip()


def synthesize(text: str, timeout: float = 180.0) -> tuple[bytes, float]:
    """Return (wav_bytes, seconds_taken). Raises on failure."""
    speak = spellable(text)
    if not speak:
        raise ValueError("nothing left to speak after normalisation")

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{config.TTS_URL}/v1/audio/speech",
            json={
                "model": "chatterbox",
                "voice": config.TTS_VOICE,
                "input": speak,
                "response_format": config.TTS_FORMAT,
            },
        )
        resp.raise_for_status()
        audio = resp.content
    return audio, time.perf_counter() - t0


def healthy() -> dict:
    """Health probe used by preflight."""
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(f"{config.TTS_URL}/health")
            resp.raise_for_status()
            return {"ok": True, **resp.json()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def duration_seconds(wav_bytes: bytes) -> float:
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return len(data) / sr
    except Exception:  # noqa: BLE001
        return 0.0


# Chatterbox generates at roughly 0.6-0.9x real time, so synthesising a whole
# reply before playing any of it leaves the user waiting silently. Splitting on
# sentence boundaries lets the first clause start playing while the rest is
# still being generated.
#
# Chunks must also stay SMALL. A long opening chunk plays out faster than the
# next one can be generated, and the speakers run dry mid-reply - the pipeline
# only helps while generation stays ahead of playback.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\.\.\.)\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+|\s+(?=(?:and|but|so|because|then)\s)")

MAX_CHUNK_CHARS = 110
MIN_CHUNK_CHARS = 24


def _split_long(part: str, max_chars: int) -> list[str]:
    """Break an over-long sentence at commas/conjunctions, then by words."""
    if len(part) <= max_chars:
        return [part]
    pieces = [p.strip() for p in _CLAUSE_END.split(part) if p and p.strip()]
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            out.append(piece)
            continue
        words = piece.split()
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > max_chars:
                out.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            out.append(current)
    return out


def sentences(
    text: str, min_chars: int = MIN_CHUNK_CHARS, max_chars: int = MAX_CHUNK_CHARS
) -> list[str]:
    """Break a reply into chunks that stay ahead of real-time playback."""
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(text or ""):
        sentence = (sentence or "").strip()
        if sentence:
            parts.extend(_split_long(sentence, max_chars))

    chunks: list[str] = []
    for part in parts:
        # Merge a stub into the previous chunk only if that stays within budget.
        if (
            chunks
            and len(part) < min_chars
            and len(chunks[-1]) + 1 + len(part) <= max_chars
        ):
            chunks[-1] = f"{chunks[-1]} {part}"
        else:
            chunks.append(part)
    return chunks


def synth_stream(text: str):
    """Yield (chunk_text, wav_bytes) so playback can start on the first clause."""
    for chunk in (sentences(text) or [text]):
        try:
            audio, _ = synthesize(chunk)
            yield chunk, audio
        except Exception as exc:  # noqa: BLE001 - skip a bad chunk, keep going
            print(f"  [tts] chunk failed ({exc}); skipping", flush=True)
