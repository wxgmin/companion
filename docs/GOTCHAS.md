# Gotchas

Every one of these cost real debugging time. They are the reason several steps
in this project look unusual, and most of them fail *quietly* rather than
loudly - which is what makes them expensive.

---

## Environment

### 1. `pip install torch` gives you the CPU wheel on Windows

`uv sync` / `pip install` resolve to `torch ...+cpu` from PyPI, and then
`torch.cuda.is_available()` is `False` with no error at install time.

Worse: installing CUDA torch *before* a `requirements.txt` does not survive,
because anything listing `torchaudio` re-resolves and pulls CPU torch back in.

**Install CUDA torch last, from the CUDA index:**

```powershell
uv pip install --python <venv>\Scripts\python.exe --reinstall torch torchaudio torchcodec `
  --index-url https://download.pytorch.org/whl/cu126
```

This bit three separate times in this project (`chatterbox-tts`,
GPT-SoVITS, WeClone). `setup.ps1` encodes the correct order.

### 2. Python 3.14 cannot run any of this

Nothing in the stack supports it (torch, numpy<2, numba, librosa,
pyopenjtalk). Use 3.11 or 3.12.

### 3. FFmpeg must be the *shared* build

torchaudio 2.9+ decodes through torchcodec, which loads FFmpeg's shared
libraries at runtime. winget's default (`Gyan.FFmpeg`) is **static** - it ships
`ffmpeg.exe` but no `avcodec` DLLs, so every reference-audio load dies:

```
Could not load libtorchcodec ... libtorchcodec_core9.dll (or one of its dependencies)
```

```powershell
winget install BtbN.FFmpeg.GPL.Shared.9.0
```

### 4. CTranslate2 needs the CUDA 12 DLLs

If you switch speech-to-text to GPU, `cublas64_12.dll` must be reachable.
Ship it on PATH or the load fails. This project runs STT on CPU by default,
which sidesteps it entirely - and costs nothing that matters.

---

## Data integrity

### 5. Moving a HuggingFace cache silently destroys it

HF caches keep real bytes in `blobs/` and **symlinks** into `snapshots/`.
`robocopy /MOVE` dereferences those links into **zero-byte placeholders**.

Every `Test-Path` still passes. A spot-check misses it completely. The failure
surfaces much later, as a parse error on an empty `config.json`, or:

```
RuntimeError: Unable to open file 'model.bin' in model '...'
```

Repair is cheap because the blobs survive - HuggingFace relinks rather than
re-downloading:

```python
from huggingface_hub import snapshot_download
snapshot_download("mobiuslabsgmbh/faster-whisper-large-v3-turbo", cache_dir=MODEL_DIR)
```

**Never `robocopy /MOVE` an HF cache.** And verify by *loading* the model, never
by checking that files exist.

### 6. faster-whisper needs the flat snapshot directory

The model lives at `.../models--<org>--<name>/snapshots/<rev>/`, but
faster-whisper wants a directory containing `model.bin` directly. Pointing it at
the cache root fails with `Unable to open file 'model.bin'`. `config.py`
resolves the snapshot automatically.

---

## Ollama

### 7. `keep_alive: -1` starves every other GPU model

A 27B at Q4 holds ~15 GB. At `keep_alive: -1` it **never releases**
(`UNTIL: Forever`), so any other GPU model fails to allocate. This presents as a
bare `No available memory for the cache blocks` while `nvidia-smi` shows most of
the card free - so it reads like a model-size problem when it is a pinning
problem.

Use `10m`, and free the GPU on demand with `ollama stop <model>`.

### 8. Thinking mode triples time-to-first-token

Qwen3.8 defaults to emitting a reasoning trace into a separate `thinking` field
and leaves `content` empty until it finishes. Pass `think: false` explicitly for
conversation.

Note this model uses Ollama's `RENDERER qwen3.5` / `PARSER qwen3.5` rather than
a Jinja template, so `ollama show --template` prints a bare `{{ .Prompt }}` and
looks broken when it is not.

### 9. Model tags lie about what is inside

Two vLLM-Omni image tags looked interchangeable and were not: `latest` was
broken (`No module named vllm.entrypoints.launchers`) and an older tagged
release predated the model entirely. Only one worked. Test the tag you actually
ship.

---

## Persona pipeline

### 10. WeClone clones **you** by default

In its state machine, `is_sender=1` is the response that gets trained on:

| | `is_sender` | becomes |
|---|---|---|
| Her messages | **1** | assistant (target) |
| Your messages | **0** | user (context) |

Getting this backwards produces a model that talks like you. The output still
looks plausible, so it is easy to ship by accident.
`scripts/verify_role_inversion.py` traces every assistant turn back to a real
message to catch it.

### 11. WeClone cannot import on Windows at all

`offline_infer.py` imports `vllm` at module level, and vLLM has no Windows
wheels - which makes the whole `qa_generator` chain unimportable. The fix makes
the heavy imports optional so the data pipeline runs natively.

### 12. WeClone joins consecutive messages inconsistently

Usually newline-joined, sometimes concatenated with no separator at all, which
produces welds like `"the video?yes i watched it like 4 times"`. Normalise all
joins.

### 13. Chinese artefacts leak into English output

When a thread opens with her message, WeClone injects a synthetic opener:

```
<begin_chat>你应该说：you forgot to call me back last night</begin_chat>
```

Strip the leading user turn.

### 14. Call events get learned as replies

Instagram exports record "started a video chat" / "chat ended" as messages. Left
in, the model learns to answer with them. Strip them per line.

### 15. WeClone prints emoji to a cp1252 console

The banner contains emoji and crashes with `UnicodeEncodeError`. Always:

```powershell
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
```

---

## Runtime

### 16. The persona model invents Unix paths on Windows

With no environment facts in the prompt, the model produced
`/root/Desktop/from_her.txt` on a Windows machine. Hermes could not create it,
**still exited 0**, and the persona reported success anyway - a hallucinated
completion that looks identical to a real one.

`config.ENVIRONMENT` is prepended to both the system prompt and every delegated
task so the model is told the real OS and paths rather than guessing.

### 17. A wrong interpreter looks like a broken service

Restarting the TTS server with the system Python instead of its venv takes the
service down with an import error that reads as a dependency problem. Check
which interpreter is running before debugging the code.

### 18. An orphaned server holds VRAM invisibly

A leftover `llama-server` held ~20 GB, which made a model spill 34% onto the CPU.
The symptom was "this model is too slow", not "something else is using the GPU".
Always check `nvidia-smi` for processes you did not expect.

---

## General rule

Almost everything above failed **silently** - a zero-byte file that passes
`Test-Path`, a model that spills to CPU without erroring, an agent that exits 0
after doing nothing.

So: verify by exercising the real path and reading ground truth (load the model,
check the file on disk, read the timing report), not by checking that the command
returned without an error.
