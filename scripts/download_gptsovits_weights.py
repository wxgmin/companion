"""Download only the GPT-SoVITS pretrained weights we actually need.

The upstream installer grabs pretrained_models.zip (~2GB) plus G2PWModel (Chinese
grapheme-to-phoneme, ~600MB) and the open_jtalk dictionary (Japanese). For an
English-only voice clone we need none of the CJK frontend assets, so we pull the
exact files referenced by GPT_SoVITS/configs/tts_infer.yaml `custom:` block:

    version: v2
    bert_base_path:      chinese-roberta-wwm-ext-large
    cnhuhbert_base_path: chinese-hubert-base
    t2s_weights_path:    gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
    vits_weights_path:   gsv-v2final-pretrained/s2G2333k.pth

(chinese-roberta-wwm-ext-large is the text encoder and chinese-hubert-base is the
SSL feature extractor -- both are used regardless of the language being spoken,
so they are not optional despite the names.)
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(r"C:\Users\Waiz\ai-companion\GPT-SoVITS\GPT_SoVITS\pretrained_models")

PATTERNS = [
    "chinese-roberta-wwm-ext-large/*",
    "chinese-hubert-base/*",
    "gsv-v2final-pretrained/*",
]


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading v2 pretrained weights into {TARGET}")
    snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        allow_patterns=PATTERNS,
        local_dir=str(TARGET),
        max_workers=4,
    )

    print("\n=== result ===")
    total = 0
    for path in sorted(TARGET.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            if size > 1024 * 1024:
                print(f"  {size / 1024 / 1024:8.1f} MB  {path.relative_to(TARGET)}")
    print(f"  total: {total / 1024 / 1024:.0f} MB")

    required = [
        TARGET / "chinese-roberta-wwm-ext-large" / "config.json",
        TARGET / "chinese-hubert-base" / "config.json",
        TARGET / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        TARGET / "gsv-v2final-pretrained" / "s2G2333k.pth",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\nMISSING REQUIRED FILES:")
        for m in missing:
            print(f"  {m}")
        return 1
    print("\nAll required v2 weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
