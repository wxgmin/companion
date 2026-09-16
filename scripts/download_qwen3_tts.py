"""Download Qwen3-TTS clone models into the labelled model store.

Why these two sizes: Qwen3-TTS publishes speaker-similarity (SIM) numbers, and
the SMALLER 0.6B model actually scores higher on English than the 1.7B:

    Qwen3-TTS-12Hz-0.6B-Base   SIM 0.829   WER 0.836
    Qwen3-TTS-12Hz-1.7B-Base   SIM 0.775   WER 0.934
    MiniMax                    SIM 0.756
    ElevenLabs                 SIM 0.613

Both are downloaded so they can be compared on the real reference clip rather
than trusting the table.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import QWEN3_TTS  # noqa: E402

MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
]


def main() -> int:
    QWEN3_TTS.mkdir(parents=True, exist_ok=True)

    for repo in MODELS:
        name = repo.split("/")[-1]
        dest = QWEN3_TTS / name
        print(f"\n=== {repo} ===")
        if dest.exists() and any(dest.iterdir()):
            size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            print(f"  already present ({size / 1024 / 1024:.0f} MB), skipping")
            continue
        print(f"  downloading to {dest}")
        snapshot_download(repo_id=repo, local_dir=str(dest))
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(f"  done: {size / 1024 / 1024:.0f} MB")

    print("\n=== store contents ===")
    total = 0
    for d in sorted(QWEN3_TTS.iterdir()):
        if d.is_dir():
            s = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            total += s
            print(f"  {s / 1024 / 1024:9.0f} MB  {d.name}")
    print(f"  {total / 1024 / 1024:9.0f} MB  TOTAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
