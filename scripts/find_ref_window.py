"""Find the best 3-10s reference window inside a longer clip.

GPT-SoVITS hard-rejects reference audio outside 3.0-10.0s, so a 62s clip must be
trimmed. Picking the window well matters more than any other single choice here:
a window that starts mid-word, contains a pause, or spans two speakers clones badly.

This transcribes the clip with word-level timestamps, groups speech into
contiguous runs, then proposes windows that:
  * land inside the 3.0-10.0s gate (targeting ~6-9s for more conditioning audio)
  * START and END on a segment boundary, so no word is cut in half
  * contain the most speech and the fewest internal gaps

Usage:
    python find_ref_window.py AUDIO [--lang en] [--min 6] [--max 9.5]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WHISPER as MODEL_STORE  # noqa: E402


def fmt(t: float) -> str:
    return f"{t:6.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lang", default=None, help="None = auto-detect")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute", default=None, help="defaults per device")
    ap.add_argument("--min", type=float, default=6.0)
    ap.add_argument("--max", type=float, default=9.5)
    ap.add_argument("--gap", type=float, default=0.7, help="max silence inside one run")
    args = ap.parse_args()

    src = Path(args.audio)
    if not src.exists():
        print(f"not found: {src}")
        return 2

    import time

    from faster_whisper import WhisperModel

    compute = args.compute or ("float16" if args.device == "cuda" else "int8")
    print(f"loading whisper ({args.model}, {args.device}/{compute})...")
    t0 = time.time()
    model = WhisperModel(
        args.model, device=args.device, compute_type=compute, download_root=MODEL_STORE
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    segments, info = model.transcribe(
        str(src),
        language=args.lang,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    segs = list(segments)
    elapsed = time.time() - t1
    print(f"  transcribed {info.duration:.1f}s of audio in {elapsed:.2f}s "
          f"({info.duration / max(elapsed, 1e-6):.1f}x realtime)")

    print(f"\ndetected language: {info.language} (p={info.language_probability:.2f})")
    print(f"duration: {info.duration:.2f}s\n")
    print("=== segments ===")
    for s in segs:
        print(f"  [{fmt(s.start)} - {fmt(s.end)}] ({s.end - s.start:5.2f}s)  {s.text.strip()}")

    # group into contiguous runs
    runs, cur = [], []
    for s in segs:
        if cur and (s.start - cur[-1].end) > args.gap:
            runs.append(cur)
            cur = []
        cur.append(s)
    if cur:
        runs.append(cur)

    print(f"\n=== {len(runs)} speech run(s) ===")
    for i, r in enumerate(runs):
        speech = sum(x.end - x.start for x in r)
        print(
            f"  run {i}: {fmt(r[0].start)} -> {fmt(r[-1].end)} "
            f"span {r[-1].end - r[0].start:6.2f}s speaking {speech:6.2f}s ({len(r)} segs)"
        )

    # enumerate candidate windows aligned to segment boundaries
    cands = []
    for r in runs:
        for i in range(len(r)):
            for j in range(i, len(r)):
                start, end = r[i].start, r[j].end
                dur = end - start
                if dur < args.min:
                    continue
                if dur > args.max:
                    break
                chunk = r[i : j + 1]
                speech = sum(x.end - x.start for x in chunk)
                gap = dur - speech
                words = sum(len(x.text.split()) for x in chunk)
                text = " ".join(x.text.strip() for x in chunk).strip()
                # prefer: dense speech, low silence, decent word count,
                # and duration close to ~7.5s
                score = (speech / dur) * 2.0 + min(words, 30) / 30.0 - gap * 0.15
                score -= abs(dur - 7.5) * 0.05
                cands.append((score, start, end, dur, speech, words, gap, text))

    cands.sort(key=lambda c: -c[0])

    print(f"\n=== top candidate windows ({args.min}-{args.max}s) ===")
    if not cands:
        print("  NONE FOUND -- no speech run is long enough in that duration range.")
        return 1

    for k, (score, start, end, dur, speech, words, gap, text) in enumerate(cands[:8], 1):
        print(f"\n  #{k}  score {score:.3f}")
        print(f"      start={start:.2f}s  end={end:.2f}s  dur={dur:.2f}s")
        print(f"      speech={speech:.2f}s  silence={gap:.2f}s  words={words}")
        print(f"      text: {text[:220]}")

    b = cands[0]
    print("\n=== BEST ===")
    print(f"  ffmpeg -ss {b[1]:.2f} -t {b[3]:.2f}")
    print(f"  transcript: {b[7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
