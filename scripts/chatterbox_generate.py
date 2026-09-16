"""Generate speech with Chatterbox (Resemble AI) using HER reference clip.

Chatterbox is the community's top pick for raw clone fidelity (5/5 in
hands-on testing, 63.75% blind preference vs ElevenLabs) and is MIT licensed,
tiny (~5 GB) and fast (RTF ~0.5). Its weakness is emotional range: there are no
emotion tags, only a single `exaggeration` knob (0.0-2.0), and reviewers
describe the default delivery as "an unexpressive human".

So this script deliberately sweeps `exaggeration` to find whether the knob
actually buys usable emotional range on HER voice.

Usage:
    python chatterbox_generate.py --ref REF.wav --text "..." --out OUT.wav \
        [--exaggeration 0.5] [--variant base|turbo|multilingual]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODEL_STORE  # noqa: E402

# keep model downloads inside the labelled store
import os

os.environ.setdefault("HF_HOME", str(MODEL_STORE / "cache" / "huggingface"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference audio (her voice)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exaggeration", type=float, default=0.5,
                    help="0.0-2.0; ~0.6 default, ~0.8 for strong emotion")
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--variant", default="base",
                    choices=["base", "turbo", "multilingual"])
    ap.add_argument("--language", default="en", help="only for --variant multilingual")
    args = ap.parse_args()

    import soundfile as sf
    import torch

    ref = Path(args.ref)
    if not ref.exists():
        print(f"reference not found: {ref}")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")
    print(f"variant: {args.variant}")
    print(f"ref    : {ref.name}")
    print(f"exagg  : {args.exaggeration}  (cfg {args.cfg_weight})")

    t0 = time.time()
    if args.variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as _C

        model = _C.from_pretrained(device=device)
    elif args.variant == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as _C

        model = _C.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS as _C

        model = _C.from_pretrained(device=device)
    print(f"loaded in {time.time() - t0:.1f}s  (sr={model.sr})")

    kwargs = {"audio_prompt_path": str(ref)}
    if args.variant == "multilingual":
        kwargs["language_id"] = args.language
    else:
        kwargs["exaggeration"] = args.exaggeration
        kwargs["cfg_weight"] = args.cfg_weight

    t1 = time.time()
    wav = model.generate(args.text, **kwargs)
    elapsed = time.time() - t1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # soundfile, not torchaudio.save: torchaudio 2.11 routes saving through
    # torchcodec, which isn't installed (and would be another FFmpeg-shared
    # dependency on Windows).
    sf.write(str(out), wav.squeeze().cpu().float().numpy(), model.sr)

    dur = wav.shape[-1] / model.sr
    print(f"\nOK in {elapsed:.1f}s -> {out}")
    print(f"audio: {model.sr} Hz, {dur:.2f}s  (RTF {elapsed / max(dur, 1e-6):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
