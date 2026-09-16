"""Download the newer GPT-SoVITS model versions (v2Pro / v2ProPlus / v4).

We started on v2, the oldest version, which is the weakest at matching a
reference timbre. v2Pro and v4 are noticeably better at "sounds like that
person", which is exactly the complaint being addressed.

Files, per GPT_SoVITS/configs/tts_infer.yaml:

    v2Pro      t2s: s1v3.ckpt            vits: v2Pro/s2Gv2Pro.pth
    v2ProPlus  t2s: s1v3.ckpt            vits: v2Pro/s2Gv2ProPlus.pth
    v4         t2s: s1v3.ckpt            vits: gsv-v4-pretrained/s2Gv4.pth
                                                                  + vocoder.pth

s1v3.ckpt is the shared GPT/T2S stage. The `sv` speaker-verification model is
also pulled because v2Pro uses speaker embeddings to improve similarity.

Downloads land in the LocalLLMs store; the junction at
GPT-SoVITS/GPT_SoVITS/pretrained_models makes them appear at the path upstream
expects.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import GPT_SOVITS_WEIGHTS as TARGET  # noqa: E402

PATTERNS = [
    "s1v3.ckpt",
    "v2Pro/*",
    "gsv-v4-pretrained/*",
    "sv/*",
]


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading v2Pro / v2ProPlus / v4 weights into {TARGET}")
    snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        allow_patterns=PATTERNS,
        local_dir=str(TARGET),
        max_workers=4,
    )

    print("\n=== present ===")
    total = 0
    for p in sorted(TARGET.rglob("*")):
        if p.is_file() and p.suffix in (".pth", ".ckpt"):
            size = p.stat().st_size
            total += size
            print(f"  {size / 1024 / 1024:8.1f} MB  {p.relative_to(TARGET)}")
    print(f"  total {total / 1024 / 1024:.0f} MB")

    required = [
        TARGET / "s1v3.ckpt",
        TARGET / "v2Pro" / "s2Gv2Pro.pth",
        TARGET / "v2Pro" / "s2Gv2ProPlus.pth",
        TARGET / "gsv-v4-pretrained" / "s2Gv4.pth",
        TARGET / "gsv-v4-pretrained" / "vocoder.pth",
    ]
    missing = [str(p.relative_to(TARGET)) for p in required if not p.exists()]
    if missing:
        print("\nMISSING:")
        for m in missing:
            print(f"  {m}")
        return 1
    print("\nAll newer-version weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
