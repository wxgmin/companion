"""Transcribe a clip exactly, for use as GPT-SoVITS `prompt_text`.

`prompt_text` must match the reference audio word for word -- it conditions the
voice, and a transcript that disagrees with the audio degrades output. A single
decode is not trustworthy enough for that, so this runs the same audio through
two different decode configurations (different compute type AND different
decoding strategy) and shows both, plus a word-level diff when they disagree.

Disagreements are the thing to look at: they mark words that need a human ear.

Usage:
    python transcribe_clip.py CLIP.wav [CLIP2.wav ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WHISPER as MODEL_STORE  # noqa: E402


def run(path: Path, device: str, compute: str, beam: int, temperature: float):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3-turbo", device=device, compute_type=compute, download_root=MODEL_STORE
    )
    segments, info = model.transcribe(
        str(path),
        language="en",
        beam_size=beam,
        temperature=temperature,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, info, segs


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    for arg in sys.argv[1:]:
        p = Path(arg)
        print("\n" + "=" * 72)
        print(f"{p.name}")
        print("=" * 72)

        results = {}
        # Two genuinely different decodes
        try:
            t1, _, _ = run(p, "cuda", "float16", beam=5, temperature=0.0)
            results["cuda/fp16/beam5"] = t1
        except Exception as exc:  # noqa: BLE001
            print(f"  cuda decode failed: {exc}")

        try:
            t2, _, _ = run(p, "cpu", "int8", beam=1, temperature=0.4)
            results["cpu/int8/beam1"] = t2
        except Exception as exc:  # noqa: BLE001
            print(f"  cpu decode failed: {exc}")

        for label, text in results.items():
            print(f"\n  [{label}]")
            print(f"  {text}")

        if len(results) == 2:
            a, b = results.values()
            wa, wb = a.lower().split(), b.lower().split()
            import difflib

            diff = [
                d
                for d in difflib.ndiff(wa, wb)
                if d.startswith("- ") or d.startswith("+ ")
            ]
            print("\n  AGREEMENT:", "identical" if a.strip() == b.strip() else "DIFFERS")
            if diff:
                print("  word diffs (- cuda, + cpu):")
                for d in diff:
                    print(f"    {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
