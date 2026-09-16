"""Generate one GPT-SoVITS config per model version, for A/B testing.

TTS_Config picks the `custom` block (TTS.py:318), so each file only needs that
one block. A full restart per version is required rather than the
/set_sovits_weights endpoint, because init_vits_weights() does NOT clear
`prompt_cache` -- switching weights live would leave reference embeddings
computed by the previous model and make the comparison meaningless.

Versions and their weights (paths resolve through the LocalLLMs junction):
    v2          t2s gsv-v2final-pretrained/s1bert25hz-...ckpt   vits gsv-v2final-pretrained/s2G2333k.pth
    v2Pro       t2s s1v3.ckpt                                   vits v2Pro/s2Gv2Pro.pth
    v2ProPlus   t2s s1v3.ckpt                                   vits v2Pro/s2Gv2ProPlus.pth
    v4          t2s s1v3.ckpt                                   vits gsv-v4-pretrained/s2Gv4.pth
"""

import sys
from pathlib import Path

CONFIG_DIR = Path(r"C:\Users\Waiz\ai-companion\GPT-SoVITS\GPT_SoVITS\configs")
PM = "GPT_SoVITS/pretrained_models"

BERT = f"{PM}/chinese-roberta-wwm-ext-large"
HUBERT = f"{PM}/chinese-hubert-base"

VERSIONS = {
    "v2": (f"{PM}/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
           f"{PM}/gsv-v2final-pretrained/s2G2333k.pth"),
    "v2Pro": (f"{PM}/s1v3.ckpt", f"{PM}/v2Pro/s2Gv2Pro.pth"),
    "v2ProPlus": (f"{PM}/s1v3.ckpt", f"{PM}/v2Pro/s2Gv2ProPlus.pth"),
    "v4": (f"{PM}/s1v3.ckpt", f"{PM}/gsv-v4-pretrained/s2Gv4.pth"),
}

TEMPLATE = """custom:
  bert_base_path: {bert}
  cnhuhbert_base_path: {hubert}
  device: cuda
  is_half: true
  t2s_weights_path: {t2s}
  vits_weights_path: {vits}
  version: {version}
"""


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for version, (t2s, vits) in VERSIONS.items():
        out = CONFIG_DIR / f"tts_infer_{version}.yaml"
        out.write_text(
            TEMPLATE.format(bert=BERT, hubert=HUBERT, t2s=t2s, vits=vits, version=version),
            encoding="utf-8",
        )
        print(f"  wrote {out.name}  (version={version})")

        # verify the weights actually resolve through the junction
        repo = CONFIG_DIR.parent.parent
        for label, rel in (("t2s", t2s), ("vits", vits)):
            p = repo / rel
            status = "OK" if p.exists() else "MISSING"
            if status == "MISSING":
                print(f"      !! {label}: {rel}  -> {status}")
                return 1
    print("\nall version configs written and weights verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
