"""Single source of truth for where models live on disk.

Why this exists: five scripts used to hardcode absolute model paths inside
C:\\Users\\Waiz\\LocalLLMs. Reorganising the store into labelled folders broke
all of them at once. Import from here instead, and a future move is a one-line
change.

    from paths import WHISPER, GPT_SOVITS_WEIGHTS
"""

from pathlib import Path

# --- project root ----------------------------------------------------------
PROJECT = Path(r"C:\Users\Waiz\ai-companion")

# --- model store -----------------------------------------------------------
MODEL_STORE = Path(r"C:\Users\Waiz\LocalLLMs\Voice Models")

_TTS = MODEL_STORE / "TTS - Voice Cloning"
GPT_SOVITS_WEIGHTS = _TTS / "GPT-SoVITS" / "pretrained_models"
UVR5_WEIGHTS = _TTS / "GPT-SoVITS" / "uvr5_weights"
QWEN3_TTS = _TTS / "Qwen3-TTS"

_ASR = MODEL_STORE / "ASR - Speech to Text"
WHISPER = _ASR / "faster-whisper-large-v3-turbo"

HF_CACHE = MODEL_STORE / "cache" / "huggingface"

# --- repos -----------------------------------------------------------------
GPT_SOVITS_REPO = PROJECT / "GPT-SoVITS"
OLV_REPO = PROJECT / "Open-LLM-VTuber"
WECLONE_REPO = PROJECT / "WeClone"
QWEN3_TTS_REPO = PROJECT / "qwen3-tts"

# --- data ------------------------------------------------------------------
DATA = PROJECT / "data"
VOICE = DATA / "voice"
CORPUS = DATA / "corpus"


def ensure_hf_cache() -> None:
    """Point HuggingFace at the store so downloads never scatter."""
    import os

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))
