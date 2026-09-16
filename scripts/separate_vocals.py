"""Isolate a vocal stem from a noisy/musical clip using GPT-SoVITS's UVR5 models.

The reference clip came from a phone video with background music and room noise.
Feeding that straight to GPT-SoVITS clones the *music* too, so the vocal has to be
separated first. This is the same tool GPT-SoVITS's own WebUI exposes under
"vocal separation".

Models (tools/uvr5/uvr5_weights):
    HP2-人声vocals+非人声instrumentals   two-stem vocals / instrumental
    HP3_all_vocals                      vocals-only extractor
    VR-DeEchoDeReverb                   removes room reverb / echo

GOTCHA: upstream notes "# 3个VR模型vocal和ins是反的" -- for these VR models the
vocal and instrumental arguments are swapped between AudioPre and
AudioPreDeEcho. We always pass KEYWORD arguments so there is no ambiguity.

Usage:
    python separate_vocals.py INPUT.wav --model HP2 --out OUTDIR
"""

import argparse
import os
import sys
from pathlib import Path

GSV = Path(r"C:\Users\Waiz\ai-companion\GPT-SoVITS")
WEIGHTS = GSV / "tools" / "uvr5" / "uvr5_weights"

# uvr5 modules import as `from lib.lib_v5...` / `from bs_roformer...`
sys.path.insert(0, str(GSV / "tools" / "uvr5"))
os.chdir(GSV)

MODELS = {
    "HP2": "HP2-人声vocals+非人声instrumentals",
    "HP3": "HP3_all_vocals",
    "DEREVERB": "VR-DeEchoDeReverb",
}


def summarize(path: Path) -> None:
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(str(path))
    dur = len(data) / sr
    rms = float(np.sqrt(np.mean(data**2))) if len(data) else 0.0
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    print(f"    {path.name}: {dur:6.2f}s  {sr} Hz  ch={data.ndim}  peak {peak:.3f}  rms {rms:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default="HP2", choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"input not found: {src}")
        return 2

    out = Path(args.out)
    vocal_dir = out / "vocal"
    ins_dir = out / "instrumental"
    vocal_dir.mkdir(parents=True, exist_ok=True)
    ins_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODELS[args.model]
    model_path = WEIGHTS / f"{model_name}.pth"
    if not model_path.exists():
        print(f"weights missing: {model_path}")
        return 3

    is_half = args.device.startswith("cuda")
    print(f"model   : {model_name}")
    print(f"device  : {args.device} (half={is_half})")
    print(f"input   : {src}")

    if args.model == "DEREVERB":
        from vr import AudioPreDeEcho

        engine = AudioPreDeEcho(
            agg=args.agg, model_path=str(model_path), device=args.device, is_half=is_half
        )
        engine._path_audio_(
            music_file=str(src),
            vocal_root=str(vocal_dir),
            ins_root=str(ins_dir),
            format="wav",
            is_hp3=False,
        )
    else:
        from vr import AudioPre

        engine = AudioPre(
            agg=args.agg, model_path=str(model_path), device=args.device, is_half=is_half
        )
        engine._path_audio_(
            music_file=str(src),
            ins_root=str(ins_dir),
            vocal_root=str(vocal_dir),
            format="wav",
            is_hp3=(args.model == "HP3"),
        )

    print("\n=== outputs ===")
    for d in (vocal_dir, ins_dir):
        files = sorted(d.glob("*.wav"))
        if not files:
            print(f"  {d.name}: (none)")
        for f in files:
            summarize(f)

    del engine
    return 0


if __name__ == "__main__":
    sys.exit(main())
