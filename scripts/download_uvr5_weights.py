"""Download the UVR5 vocal-separation models used to clean the reference clip.

GPT-SoVITS's own installer fetches these from lj1995/VoiceConversionWebUI. Only
the .pth (VR-architecture) models are used here:

  HP2-人声vocals+非人声instrumentals.pth
      Two-stem split: vocals / instrumental. This is the one that strips
      background MUSIC out of the phone video.
  HP3_all_vocals.pth
      Vocals-only extractor. Upstream's own notes say HP3 preserves the lead
      vocal marginally better than HP2, so it's kept as a comparison.
  VR-DeEchoDeReverb.pth
      Removes room reverb/echo -- valuable for a phone recording.

bs_roformer (higher SDR) is NOT published in this repo, so it is not available
without pulling weights from a third-party source.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(r"C:\Users\Waiz\ai-companion\GPT-SoVITS\tools\uvr5\uvr5_weights")

WANTED = [
    "uvr5_weights/HP2-人声vocals+非人声instrumentals.pth",
    "uvr5_weights/HP3_all_vocals.pth",
    "uvr5_weights/VR-DeEchoDeReverb.pth",
]


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(WANTED)} UVR5 models into {TARGET}")

    snapshot_download(
        repo_id="lj1995/VoiceConversionWebUI",
        allow_patterns=WANTED,
        local_dir=str(TARGET.parent.parent.parent),  # -> GPT-SoVITS/
        max_workers=3,
    )

    print("\n=== present ===")
    for p in sorted(TARGET.glob("*.pth")):
        print(f"  {p.stat().st_size / 1024 / 1024:8.1f} MB  {p.name}")

    missing = [w for w in WANTED if not (TARGET / Path(w).name).exists()]
    if missing:
        print("\nMISSING:")
        for m in missing:
            print(f"  {m}")
        return 1
    print("\nAll requested models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
