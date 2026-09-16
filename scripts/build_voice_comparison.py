"""Build one listenable comparison file across every voice model tested.

Generates a spoken label for each take using the running GPT-SoVITS server, then
concatenates: the reference clip first (the target), then every model's take,
each preceded by its own name.

All clips are resampled to a common 32 kHz mono so they can be concatenated
regardless of which model produced them (GPT-SoVITS outputs 32 kHz, Qwen3-TTS
24 kHz).

Usage:
    python build_voice_comparison.py --out COMPARISON.wav
"""

import argparse
import io
import sys
import wave
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import VOICE  # noqa: E402

TTS_URL = "http://127.0.0.1:9880/tts"
SR = 32000

# Label, path. The reference comes first so every take can be judged against it.
REF = VOICE / "reference.wav"
REF_PROMPT = "She is vibing. She was literally vibing and then when I asked her, she said no."

W = VOICE / "_work"

SETS = {
    # every engine tested so far
    "all": [
        ("This is her actual voice. The reference clip.", REF),
        ("GPT SoVITS, version two.", W / "ab" / "v2__D.wav"),
        ("GPT SoVITS, version two pro.", W / "ab" / "v2Pro__D.wav"),
        ("GPT SoVITS, version two pro plus.", W / "ab" / "v2ProPlus__D.wav"),
        ("GPT SoVITS, version four.", W / "ab" / "v4__D.wav"),
        ("Qwen three TTS, zero point six billion.", W / "ab_qwen" / "qwen0.6B__transcript.wav"),
        ("Qwen three TTS, one point seven billion.", W / "ab_qwen" / "qwen1.7B__transcript.wav"),
        ("Chatterbox. Exaggeration zero point five.", W / "ab_chatterbox" / "cb_exag0.5.wav"),
        ("Chatterbox. Exaggeration zero point eight.", W / "ab_chatterbox" / "cb_exag0.8.wav"),
        ("Chatterbox. Exaggeration one point one.", W / "ab_chatterbox" / "cb_exag1.1.wav"),
        ("Chatterbox Turbo.", W / "ab_chatterbox" / "cb_turbo.wav"),
    ],
    # just the new engine, so it can be judged against her real voice directly
    "chatterbox": [
        ("This is her actual voice. The reference clip.", REF),
        ("Chatterbox. Exaggeration zero point five. The default.", W / "ab_chatterbox" / "cb_exag0.5.wav"),
        ("Chatterbox. Exaggeration zero point eight. Pushed for emotion.", W / "ab_chatterbox" / "cb_exag0.8.wav"),
        ("Chatterbox. Exaggeration one point one. Maximum.", W / "ab_chatterbox" / "cb_exag1.1.wav"),
        ("Chatterbox Turbo. The fast variant.", W / "ab_chatterbox" / "cb_turbo.wav"),
    ],
}

MANIFEST = SETS["all"]


def tts_label(text: str) -> np.ndarray | None:
    """Ask the running GPT-SoVITS server to say the label."""
    payload = {
        "text": text,
        "text_lang": "en",
        "ref_audio_path": str(REF),
        "prompt_text": REF_PROMPT,
        "prompt_lang": "en",
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
    }
    try:
        r = requests.post(TTS_URL, json=payload, timeout=180)
        if r.status_code != 200:
            print(f"    label failed: HTTP {r.status_code}")
            return None
        data, sr = sf.read(io.BytesIO(r.content))
    except Exception as exc:  # noqa: BLE001
        print(f"    label failed: {exc}")
        return None
    return resample(data, sr)


def resample(data: np.ndarray, sr: int) -> np.ndarray:
    """-> float32 mono at SR."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa

        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SR)
    return data.astype(np.float32)


def load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data, sr = sf.read(str(path))
    return resample(data, sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="all", choices=sorted(SETS))
    ap.add_argument("--gap", type=float, default=0.55)
    ap.add_argument("--pad", type=float, default=0.12)
    args = ap.parse_args()

    manifest = SETS[args.set]

    silence = np.zeros(int(args.gap * SR), dtype=np.float32)
    pad = np.zeros(int(args.pad * SR), dtype=np.float32)

    pieces = []
    print(f"building comparison (set: {args.set}):")
    for label, path in manifest:
        audio = load(path)
        if audio is None:
            print(f"  SKIP (missing) {path.name}")
            continue

        lab = tts_label(label)
        print(f"  + {label[:52]:54s} <- {path.name}")

        if lab is not None:
            pieces += [lab, silence]
        pieces += [audio, silence]

    if not pieces:
        print("nothing to concatenate")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate([pad] + pieces)
    sf.write(str(out), combined, SR)

    total = len(combined) / SR
    print(f"\nwritten {out}  ({total:.1f}s, {out.stat().st_size / 1024:.0f} KB)")
    print("play it, then tell me which one sounds most like her")
    return 0


if __name__ == "__main__":
    sys.exit(main())
