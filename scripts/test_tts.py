"""Smoke-test the GPT-SoVITS /tts endpoint.

Posts to the local GPT-SoVITS api_v2 server and writes the returned wav.
Use this to verify the voice chain is alive before wiring up Open-LLM-VTuber.

Usage:
    python test_tts.py [--ref PATH] [--prompt-text TEXT] [--text TEXT] [--out PATH]

Defaults point at the placeholder reference shipped for pipeline testing.
"""

import argparse
import sys
import time
from pathlib import Path

import requests

DEFAULT_REF = r"C:\Users\Waiz\ai-companion\data\voice\_PLACEHOLDER_main_sample.wav"

# Must match the reference clip EXACTLY -- GPT-SoVITS conditions on this text.
DEFAULT_PROMPT_TEXT = (
    "This is a sample voice for you to just get started with because it sounds "
    "kind of cute but just make sure this doesn't have long silences."
)

DEFAULT_TEXT = "Okay, the voice server is running and this is what I sound like now."

URL = "http://127.0.0.1:9880/tts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    ap.add_argument("--prompt-lang", default="en")
    ap.add_argument("--text-lang", default="en")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", default=r"C:\Users\Waiz\ai-companion\data\voice\_tts_test_output.wav")
    args = ap.parse_args()

    if not Path(args.ref).exists():
        print(f"reference audio not found: {args.ref}")
        return 2

    payload = {
        "text": args.text,
        "text_lang": args.text_lang,
        "ref_audio_path": args.ref,
        "prompt_text": args.prompt_text,
        "prompt_lang": args.prompt_lang,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
    }

    print(f"POST {URL}")
    print(f"  ref : {args.ref}")
    print(f"  text: {args.text!r}")

    started = time.time()
    try:
        r = requests.post(URL, json=payload, timeout=600)
    except requests.exceptions.ConnectionError:
        print("\nCONNECTION REFUSED -- is api_v2.py running on port 9880?")
        return 3

    elapsed = time.time() - started

    if r.status_code != 200:
        print(f"\nHTTP {r.status_code} after {elapsed:.1f}s")
        print(r.text[:2000])
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)

    print(f"\nOK in {elapsed:.1f}s -> {out} ({len(r.content) / 1024:.0f} KB)")

    try:
        import soundfile as sf

        data, sr = sf.read(str(out))
        print(f"audio: {sr} Hz, {len(data) / sr:.2f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not introspect audio: {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
