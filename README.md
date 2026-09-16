# Companion

A local voice companion that is two things at once: a character who talks in a
specific person's cloned voice and persona, and an agent you can hand real work
to. Everything runs on your own machine - no API keys, no cloud, nothing
leaving the box.

You talk to her. She answers out loud in a voice cloned from a few seconds of
reference audio. When you ask for something that actually needs doing - a file,
a command, a search, a script - she hands it to a background agent and then
tells you what happened, still in character.

```
        your voice
            |
            v
   +--------------------+      +------------------+
   |  faster-whisper    |      |   Chatterbox     |
   |  speech to text    |      |   cloned voice   |
   +--------------------+      +------------------+
            |                           ^
            v                           |
   +--------------------------------------------+
   |            orchestrator                    |
   |  decides: just talk, or actually do work   |
   +--------------------------------------------+
            |                           ^
            v                           |
   +--------------------+      +------------------+
   |  local LLM         |<-----|  Hermes agent    |
   |  the persona       |      |  (files, shell,  |
   |                    |      |   web, code)     |
   +--------------------+      +------------------+
```

---

## What you need

- An NVIDIA GPU with **24 GB**. The measured configuration assumes that; smaller
  cards work if you drop the model or the context size.
- Windows, Linux, or macOS
- Python 3.11 or 3.12
- [Ollama](https://ollama.com/download)
- A shared FFmpeg build - `winget install --id BtbN.FFmpeg.GPL.Shared`.
  Static builds break `torchcodec` with a missing-DLL error (see
  [`docs/GOTCHAS.md`](docs/GOTCHAS.md)).
- **[Hermes](https://github.com/NousResearch/hermes)** for the task side.
  Without it she still talks, she just cannot do anything.

---

## Setup

```powershell
git clone <your-fork>
cd companion
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

That builds the Python environment, pulls the language model, and installs
Chatterbox with CUDA torch. It is safe to re-run.

Then add the two things only you can provide:

1. **Her voice** - a 3-10 second clip at `data/voice/reference.wav`.
   See [`data/voice/README.md`](data/voice/README.md).
2. **Her chats** - exports under `data/persona/`.
   See [`data/persona/README.md`](data/persona/README.md).

Finally:

```powershell
companion.cmd --voice
```

and start talking. There is no push-to-talk: speech starts a turn, and about
three quarters of a second of silence ends it. Talking over her stops her.

---

## The two halves

### The voice half

Speech is transcribed by `faster-whisper`, answered by a local model, and spoken
by **Chatterbox**, which conditions on `reference.wav` for every utterance.
Nothing about her voice is a preset - it is cloned from your clip each time the
server starts.

Chatterbox generates at roughly 0.6-0.9x real time, so the reply is split into
clause-sized chunks and played as it is produced. The first words land around
**1.8 seconds** instead of waiting ~9 seconds for a whole paragraph.

### The task half

She is not hard-coded to react to words like "open" or "search". She is given a
single `delegate_task` tool and decides for herself when something needs real
work. Conversation stays conversation; work is routed to the **Hermes** agent,
which runs the tools. Its report comes back to her, and she tells you about it
the way a person would.

```
you:  make a file on my desktop called note.txt that says hello
her:  done. it's sitting on your desktop right now, exactly as i told it.
      go check, i dare you.
```

This split is why a 27B stays conversational: the model handles the talking and
only reaches for the agent when there is actually something to do.

### Where coding work goes

Hermes is linked to **Command Code** (`cmdc`) through a skill installed into
Hermes' skills directory:

```
you -> her -> Hermes agent -> cmdc -> code changes on disk
```

Hermes decides when a job is big enough to hand to a coding agent rather than
doing it inline, and `integrations/hermes/command-code/SKILL.md` documents the
invocation it uses. Two details in there matter more than they look:

- **Print mode withholds the file and shell tools.** A `cmdc -p ...` run
  without `--yolo` (or `--tools-all`) exits `0` after explaining that it had no
  tools, which reads exactly like a successful no-op.
- **Hermes' shell ignores the process working directory.** It resolves
  `terminal.cwd` from its own config, so a delegated task writes into the home
  directory unless `TERMINAL_CWD` is set. The orchestrator sets it per
  invocation; `--in` does not do this, it only scopes session lookup.

Both failure modes are silent - the command succeeds and the work does not
happen - so `orchestrator/test_chain.py` checks ground truth on disk.

---

## Making it *her*

`data/persona/README.md` covers the mechanics. The part worth stating plainly
here:

**The pipeline trains on one speaker. Messages from you are context, never
targets.** Invert that and you get a model that talks like you - a subtle and
expensive failure, because the output still looks plausible.

`scripts/verify_role_inversion.py` checks it mechanically: it traces every
generated assistant turn back to a real message and confirms whose it was. Run
it after any change to the parsing.

All sources are merged into a single chronological timeline before training,
because the same conversation continues across platforms. Ordering is
deterministic on `(timestamp, source, original_id)`, so re-running produces
identical output.

---

## Measured budget

Numbers from a 24 GB card with the persona model resident at 64K context:

| Component | VRAM |
|---|---|
| Desktop / display | ~1,300 MiB |
| Chatterbox | 3,730 MiB |
| 27B at 64K context | ~17,000 MiB |
| **Total** | **~22,100 MiB of 24,576** |

Speech-to-text runs on the **CPU** on purpose. It costs zero VRAM, loads in
2.3s, and transcribes a 3-second clip in 0.18s - the freed 1.5 GB is worth far
more to the language model than the microseconds are to STT.

Latency, warm:

| Stage | Time |
|---|---|
| Speech to text | ~0.2s |
| Language model | ~2.9s (44 tok/s) |
| First audio | ~1.8s after the reply starts |
| Full round trip | ~4-5s |

`start.py` runs a preflight before anything starts, because the failure mode
here is quiet rather than loud: if context overflows VRAM, Ollama spills layers
to the CPU without erroring, and the only symptom is that everything is slow and
tool calls stop parsing.

---

## Configuration

Everything tunable lives in `orchestrator/config.py`. The ones you are most
likely to touch:

| Setting | Default | Notes |
|---|---|---|
| `PERSONA_MODEL` | `orcarouter/Qwen3.8-27B-Uncensored:iq4_xs` | Any Ollama tag |
| `PERSONA_NUM_CTX` | `65536` | Highest that still measured 100% on GPU |
| `PERSONA_THINK` | `0` | Keep 0 - thinking triples time-to-first-token |
| `KEEP_ALIVE` | `10m` | **Never `-1`** - it pins ~16 GB forever |
| `TTS_EXAGGERATION` | `0.8` | Higher is more emotional, less stable |
| `WHISPER_DEVICE` | `cpu` | `cuda` works but costs ~1.5 GB |
| `COMPANION_PERSONA_PROMPT` | built-in | Override to write your own character |

### Retuning for a smaller GPU

Drop the model first, then the context:

```
qwen3:14b              ~9 GB     comfortable at 64K alongside everything
qwen3:8b               ~5 GB     lots of headroom, weaker at tool calls
```

---

## Commands

```powershell
companion.cmd                      # start everything, leave it serving
companion.cmd --voice              # start everything and open the mic loop
companion.cmd --no-serve           # services only, no HTTP server
companion.cmd --reference other.wav
```

```powershell
# check the reference clip before trusting it
orchestrator\.venv\Scripts\python.exe scripts\check_ref_audio.py data\voice\reference.wav

# confirm the persona data is pointed the right way
orchestrator\.venv\Scripts\python.exe scripts\verify_role_inversion.py

# talk to the API without a microphone
curl -X POST http://127.0.0.1:8090/chat `
     -H "Content-Type: application/json" `
     -d '{\"text\":\"hey\",\"speak\":true}'
```

The server also exposes `ws://127.0.0.1:8090/ws` for a front end. Audio arrives
as a separate event from the text, so a lip-sync avatar can start moving the
moment the reply is known.

---

## Troubleshooting

**`torch.cuda.is_available()` is False.** A CPU-only torch wheel got installed.
Re-run setup, or force it:
`pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchaudio`

**She speaks but the task side does nothing.** Hermes is not on PATH. Check with
`hermes status`.

**She says a task is done when it is not.** Check that `ENVIRONMENT` in
`config.py` matches your actual machine. A model that invents Unix paths on
Windows produces tasks that fail while the agent still exits 0.

**Everything is slow all of a sudden.** VRAM has spilled to the CPU. Run
`orchestrator\.venv\Scripts\python.exe orchestrator\preflight.py` and confirm the
model is still 100% on GPU.

**Voice sounds unstable or wobbly.** The reference clip is too short, too long,
or contains a second speaker or music. See `data/voice/README.md`.

**`Unable to open file 'model.bin'`.** The HuggingFace cache has been damaged -
usually by a `robocopy /MOVE` that dereferenced the symlinks into 0-byte files.
`Test-Path` passes on every file, so it looks fine. See
[`docs/GOTCHAS.md`](docs/GOTCHAS.md).

More, with explanations: [`docs/GOTCHAS.md`](docs/GOTCHAS.md).

---

## Layout

```
companion/
  companion.cmd              one-command launcher
  orchestrator/
    config.py                every tunable in one place
    start.py                 starts the stack in dependency order
    preflight.py             VRAM and wiring checks
    llm.py                   persona model client
    tts.py                   Chatterbox client + chunked playback
    voice_loop.py            mic -> VAD -> STT -> reply -> speech
    router.py                chat vs. work, drives Hermes
    server.py                HTTP + WebSocket
    verify_stack.py          acceptance check for the running stack
    test_chain.py            orchestrator -> Hermes -> cmdc, checks files on disk
    test_voice_loop.py       deterministic turn-taking test
    measure_stack.py         staged VRAM accounting
  integrations/hermes/
    command-code/SKILL.md    teaches Hermes to delegate to cmdc
  scripts/
    setup.ps1                fresh-machine install
    build_persona.ps1        chat exports -> training data
    check_ref_audio.py       validates the voice reference
    verify_role_inversion.py proves the training direction
  data/
    voice/                   your reference audio (never committed)
    persona/                 your chat exports (never committed)
  docs/GOTCHAS.md            the failures that cost real time
```

---

## License

MIT for the code. Your voice and chat data are yours and are explicitly
git-ignored - nothing about the people involved ships with this repository.
