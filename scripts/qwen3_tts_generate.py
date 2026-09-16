"""Zero-shot voice cloning with Qwen3-TTS, for direct comparison with GPT-SoVITS.

Qwen3-TTS clones from a ~3 second reference, so the 7s clip used for GPT-SoVITS
works comfortably. Two modes:

  normal     reference audio + its exact transcript. Best quality.
  --xvector  speaker embedding only; NO transcript needed, lower quality.
             Useful because our reference has a mumbled tail.

Note: flash-attn is not installed on this machine, so we omit
`attn_implementation` and let it fall back to the built-in PyTorch attention.
Slower, same output.

Usage:
    python qwen3_tts_generate.py --size 0.6B --ref REF.wav --ref-text "..." \
        --text "..." --out OUT.wav
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import QWEN3_TTS  # noqa: E402

SIZES = {
    "0.6B": "Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen3-TTS-12Hz-1.7B-Base",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=sorted(SIZES))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xvector", action="store_true",
                    help="no transcript needed, lower similarity")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model_dir = QWEN3_TTS / SIZES[args.size]
    if not model_dir.exists():
        print(f"model not downloaded: {model_dir}")
        return 2
    ref = Path(args.ref)
    if not ref.exists():
        print(f"reference not found: {ref}")
        return 2

    print(f"model   : {model_dir.name}")
    print(f"device  : cuda (bf16)")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    kwargs = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    if args.xvector:
        print("mode    : x-vector only (no transcript)")
        if args.ref_text:
            print("          (ref_text ignored)")
    else:
        if not args.ref_text:
            print("need --ref-text unless --xvector is set")
            return 2
        print(f"ref_text: {args.ref_text[:80]}{'...' if len(args.ref_text) > 80 else ''}")
    print(f"text    : {args.text[:80]}")

    t1 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=str(ref),
        ref_text=None if args.xvector else args.ref_text,
        x_vector_only_mode=args.xvector,
        max_new_tokens=args.max_new_tokens,
        **kwargs,
    )
    elapsed = time.time() - t1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wavs[0], sr)

    dur = len(wavs[0]) / sr
    print(f"\nOK in {elapsed:.1f}s -> {out}")
    print(f"audio: {sr} Hz, {dur:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
