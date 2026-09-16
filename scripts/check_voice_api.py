"""Round-trip check: synthesize a known sentence, then transcribe it back.

A transcription endpoint that merely returns 200 proves nothing, so this
compares what came out against what went in. It also exercises the multipart
shape AIRI's hearing provider uses, which is the part most likely to be wrong.

    python scripts/check_voice_api.py
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request

BASE = "http://127.0.0.1:8092/v1"
SPOKEN = "Hey, it is me. I was just thinking about you and wondering how your day went."


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----companionboundary"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
    )
    body.write(b"Content-Type: audio/wav\r\n\r\n")
    body.write(data)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ok = True

    # --- speech ---------------------------------------------------------
    print("1. /v1/audio/speech")
    payload = (
        '{"model":"chatterbox","input":"%s","voice":"her","response_format":"wav"}'
        % SPOKEN
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/audio/speech", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        wav = resp.read()
    print(f"   -> {len(wav)} bytes, content-type {resp.headers.get('Content-Type')}")
    if len(wav) < 1000:
        print("   FAIL: audio too small")
        ok = False

    # --- transcriptions -------------------------------------------------
    print("2. /v1/audio/transcriptions")
    body, ctype = _multipart("file", "clip.wav", wav)
    req = urllib.request.Request(
        f"{BASE}/audio/transcriptions", data=body,
        headers={"Content-Type": ctype},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = resp.read().decode()
    except urllib.error.HTTPError as exc:
        print(f"   FAIL: HTTP {exc.code} {exc.read().decode()[:200]}")
        return 1

    print(f"   -> {out[:220]}")

    # Compare on letters only: TTS/ASR will differ in punctuation and casing,
    # and blaming that would be noise rather than signal.
    def norm(s: str) -> str:
        return re.sub(r"[^a-z ]", "", s.lower()).strip()

    import json

    try:
        got = norm(json.loads(out).get("text", ""))
    except (ValueError, AttributeError):
        got = norm(out)

    want = norm(SPOKEN)
    want_words = set(want.split())
    got_words = set(got.split())
    overlap = len(want_words & got_words) / max(len(want_words), 1)
    print(f"   word overlap: {overlap:.0%}")

    if overlap < 0.6:
        print("   FAIL: transcript does not resemble the spoken text")
        ok = False

    print()
    print("RESULT:", "both directions working" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
