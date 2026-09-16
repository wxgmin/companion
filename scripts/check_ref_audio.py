"""Check whether an audio file is usable as a GPT-SoVITS reference clip.

Replicates exactly what GPT-SoVITS does when you pass `ref_audio_path`:

  GPT_SoVITS/TTS_infer_pack/TTS.py :: _set_prompt_semantic
      wav16k, sr = librosa.load(ref_wav_path, sr=16000)
      if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
          raise OSError("reference audio outside 3~10s range")

  GPT_SoVITS/TTS_infer_pack/TTS.py :: _get_ref_spec
      raw_audio, raw_sr = torchaudio.load(ref_audio_path)

Both must succeed. librosa goes through libsndfile; torchaudio goes through
torchcodec/FFmpeg. A format that only one of them can read will fail at request
time, so this checks both.

Usage:
    python check_ref_audio.py FILE [FILE ...]
"""

import sys
from pathlib import Path

MIN_SAMPLES = 48000  # 3.0s @ 16k
MAX_SAMPLES = 160000  # 10.0s @ 16k


def check(path: Path) -> bool:
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print("  MISSING")
        return False

    print(f"  size: {path.stat().st_size / 1024:.1f} KB")

    ok = True

    # --- what librosa sees (this is the hard gate) ---
    try:
        import librosa
        import numpy as np

        wav16k, sr = librosa.load(str(path), sr=16000)
        n = wav16k.shape[0]
        dur = n / 16000
        print(f"  librosa : OK  {n} samples @16k = {dur:.2f}s")

        if n < MIN_SAMPLES:
            print(f"            FAIL -> too SHORT ({dur:.2f}s, need >= 3.0s)")
            ok = False
        elif n > MAX_SAMPLES:
            print(f"            FAIL -> too LONG ({dur:.2f}s, need <= 10.0s)")
            ok = False

        peak = float(np.max(np.abs(wav16k))) if n else 0.0
        rms = float(np.sqrt(np.mean(wav16k**2))) if n else 0.0
        print(f"  signal  : peak {peak:.3f}  rms {rms:.4f}")
        if rms < 0.005:
            print("            WARN -> near-silent, bad reference")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  librosa : FAIL -> {type(exc).__name__}: {str(exc)[:150]}")
        ok = False

    # --- what torchaudio sees (the spectrogram path) ---
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        print(f"  torchaud: OK  shape {tuple(wav.shape)} @ {sr} Hz")
    except Exception as exc:  # noqa: BLE001
        print(f"  torchaud: FAIL -> {type(exc).__name__}: {str(exc)[:150]}")
        ok = False

    # --- container/codec detail ---
    try:
        import soundfile as sf

        info = sf.info(str(path))
        print(f"  codec   : {info.format} / {info.subtype}, {info.samplerate} Hz, {info.channels} ch")
    except Exception as exc:  # noqa: BLE001
        print(f"  codec   : (soundfile cannot open: {type(exc).__name__})")

    print(f"  VERDICT : {'USABLE' if ok else 'NOT USABLE'}")
    return ok


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(2)
    results = [check(p) for p in paths]
    print(f"\n{sum(results)}/{len(results)} usable")
