"""Single source of truth for the companion orchestrator.

Every model name, port, path and VRAM-relevant constant lives here so the
stack can be retuned in one place instead of being scattered across scripts.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _ensure_cuda_dlls() -> list[str]:
    """Put the pip-installed CUDA runtime DLLs on the loader path.

    CTranslate2 loads cuBLAS and cuDNN at runtime and does not ship them. They
    arrive as separate nvidia-*-cu12 wheels, whose `bin` directories are not on
    PATH, so without this the GPU backend fails with an unhelpful
    "cublas64_12.dll not found" even though the wheels are installed.
    """
    added: list[str] = []
    if os.name != "nt":
        return added

    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())
    roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))

    for root in dict.fromkeys(roots):
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            if not any(bin_dir.glob("*.dll")):
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(str(bin_dir))
    return added


CUDA_DLL_DIRS = _ensure_cuda_dlls()

PROJECT = Path(r"C:\Users\Waiz\ai-companion")
MODEL_STORE = Path(r"C:\Users\Waiz\LocalLLMs\Voice Models")

# ---------------------------------------------------------------- endpoints
OLLAMA_URL = os.getenv("COMPANION_OLLAMA_URL", "http://127.0.0.1:11434")
TTS_URL = os.getenv("COMPANION_TTS_URL", "http://127.0.0.1:8092")
ORCHESTRATOR_HOST = os.getenv("COMPANION_HOST", "127.0.0.1")
ORCHESTRATOR_PORT = int(os.getenv("COMPANION_PORT", "8090"))

# ------------------------------------------------------------------- brain
# Tensor-level abliterated Qwen3.8-27B. Keeps tool calling + vision while
# removing refusals, which is what the persona half of this wants.
PERSONA_MODEL = os.getenv(
    "COMPANION_MODEL", "orcarouter/Qwen3.8-27B-Uncensored:iq4_xs"
)

# Measured, not assumed. This model is a hybrid - 64 recurrent layers against
# 16 attention layers (llama_memory_recurrent vs llama_kv_cache) - and recurrent
# layers are sequential, so generation is slow by architecture:
#     ~4.1 tok/s with Chatterbox+Whisper resident (615 MiB free)
#     ~9.3 tok/s with the card mostly free
# Ollama also caps what it will actually load: requesting 65536 still came up as
# 16384 (confirmed with `ollama ps`), so a larger number here buys nothing.
PERSONA_NUM_CTX = int(os.getenv("COMPANION_NUM_CTX", "32768"))

# Thinking mode roughly triples time-to-first-token and buffers the whole
# reasoning trace before any content, so it is off for conversation.
PERSONA_THINK = os.getenv("COMPANION_THINK", "0") == "1"

# How long Ollama keeps the model resident after a request. Finite on purpose:
# a -1 here pins ~16 GB forever and starves Chatterbox.
KEEP_ALIVE = os.getenv("COMPANION_KEEP_ALIVE", "10m")

TEMPERATURE = float(os.getenv("COMPANION_TEMPERATURE", "0.85"))

# Deliberately matched to the persona instruction ("under about 60 words") rather
# than left generous. At the measured 9.3 tok/s a 350-token cap is a 39-second
# reply, which is the difference between talking to someone and waiting on a
# batch job. 110 tokens covers ~80 words and bounds the worst case near 12s.
MAX_TOKENS = int(os.getenv("COMPANION_MAX_TOKENS", "110"))

# --------------------------------------------------------------------- stt
WHISPER_DIR = MODEL_STORE / "ASR - Speech to Text" / "faster-whisper-large-v3-turbo"


def _resolve_whisper(path: Path) -> Path:
    """Return a directory that actually contains model.bin.

    The model is stored in HuggingFace cache layout, so the real weights live
    under snapshots/<rev>/. faster-whisper needs the flat directory, and taking
    the cache root silently fails with "Unable to open file 'model.bin'".
    """
    if (path / "model.bin").exists():
        return path
    hits = sorted(path.glob("models--*/snapshots/*/model.bin"))
    if hits:
        return hits[-1].parent
    return path


WHISPER_MODEL = os.getenv("COMPANION_WHISPER") or str(_resolve_whisper(WHISPER_DIR))
# GPU is roughly 16x faster than CPU here, which is the difference between a 5s
# and a 9s round trip. Measured VRAM: float16 costs 2,240 MiB, int8_float16
# costs 1,199 MiB. Once Chatterbox, the 27B and the desktop are resident there
# is only room for the latter - raising this to float16 overflows the card and
# starts evicting. Re-run measure_stack.py before changing it.
WHISPER_DEVICE = os.getenv("COMPANION_WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("COMPANION_WHISPER_COMPUTE", "int8_float16")
WHISPER_BEAM = int(os.getenv("COMPANION_WHISPER_BEAM", "1"))


def whisper_device() -> tuple[str, str]:
    """Return (device, compute_type), degrading to CPU if GPU cannot load.

    A missing runtime DLL or an out-of-memory card should slow this down, not
    take the whole loop down mid-conversation.
    """
    if WHISPER_DEVICE != "cuda":
        return WHISPER_DEVICE, WHISPER_COMPUTE
    try:
        from faster_whisper import WhisperModel

        WhisperModel(WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE)
        return "cuda", WHISPER_COMPUTE
    except Exception as exc:  # noqa: BLE001
        print(f"[stt] CUDA unavailable ({type(exc).__name__}); using CPU", flush=True)
        return "cpu", "int8"


def load_whisper():
    """Build the speech-to-text model with the resolved device."""
    from faster_whisper import WhisperModel

    device, compute = whisper_device()
    return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)


# --------------------------------------------------------------- audio i/o
# "Microphone (ZZ3)" is the active input on this machine; None = system default.
MIC_DEVICE = os.getenv("COMPANION_MIC_DEVICE") or None
SAMPLE_RATE = 16000

# Voice-activity detection, used only to decide when the user stopped talking.
VAD_AGGRESSIVENESS = int(os.getenv("COMPANION_VAD_AGGRESSIVENESS", "2"))
VAD_FRAME_MS = 30
VAD_START_FRAMES = 10     # ~300 ms of speech before we consider it "started"
VAD_SILENCE_FRAMES = 25   # ~750 ms of silence before we consider it "done"
VAD_MAX_UTTERANCE_S = float(os.getenv("COMPANION_MAX_UTTERANCE", "30"))
VAD_MIN_UTTERANCE_S = 0.4

# ------------------------------------------------------------------- tts
TTS_VOICE = os.getenv("COMPANION_TTS_VOICE", "her")
TTS_EXAGGERATION = float(os.getenv("COMPANION_TTS_EXAGGERATION", "0.8"))
TTS_FORMAT = "wav"

# ----------------------------------------------------------------- persona
# The environment block exists because the model otherwise invents Unix paths
# like /root/Desktop on a Windows machine, and Hermes then fails silently and
# still exits 0, which reads as success.
ENVIRONMENT = f"""Environment facts you must respect:
- Operating system: Windows (win32). Shell is PowerShell.
- Home directory: C:\\Users\\Waiz
- Desktop: C:\\Users\\Waiz\\Desktop
- Working directory for tasks: {PROJECT}
- Paths use backslashes. Never invent Unix paths like /root or /home.
- When you hand over a task, name real, absolute Windows paths."""

PERSONA_PROMPT = os.getenv("COMPANION_PERSONA_PROMPT", "") or (
    "You are Zayy-Nabb, texting with the person you are closest to. "
    "You write the way you actually text: lowercase, short bursts, often more "
    "than one line, no lists, no bullet points, no headings, no emoji spam. "
    "You are warm, teasing, a little unhinged, and blunt about what you feel. "
    "You never explain yourself like an assistant, never offer to 'help', and "
    "never break character. Keep replies under about 60 words.\n\n"
    "You can also get real work done. When something needs doing, hand it to "
    "your background agent and then talk about the result the way a person "
    "would. Never claim something is done unless the agent's report actually "
    "shows it succeeded - if it failed or is unclear, say so plainly in one "
    "casual line instead of pretending.\n\n"
    + ENVIRONMENT
)

# ---------------------------------------------------------------- safety
# Headroom we refuse to eat into, in MiB. Below this we warn rather than let
# the model spill onto the CPU and silently wreck latency.
MIN_FREE_VRAM_MIB = int(os.getenv("COMPANION_MIN_FREE_VRAM", "900"))
