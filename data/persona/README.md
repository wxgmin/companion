# Chat history goes here

This directory is where **your** chat exports live. Nothing in here is
committed to git.

## What to put here

Raw exports from wherever you actually talked:

```
data/persona/
  whatsapp/
    _chat.txt              # WhatsApp "Export chat" output (without media)
  instagram/
    account1/messages.html # Instagram "Download your information" HTML
    account2/messages.html
```

Subfolder names do not matter. What matters is that the files are the
**original exports**, unedited.

## The one rule that matters

The pipeline trains on one speaker only. In every export, the person you want
the companion to become must be identifiable, and messages from *you* must be
identifiable as yours.

- WhatsApp exports label every line with a sender name, so it works as-is.
- Instagram exports carry the account owner's name, which is why the parser
  needs `PERSONA_NAMES` set (**see `orchestrator/parse.py`** or run
  `scripts/build_persona.ps1` with `--persona-names`).

Getting this backwards produces a model that talks like you instead of like
her. `scripts/verify_role_inversion.py` exists to catch exactly that: it walks
every generated training example and checks that the assistant turns trace back
to real messages from her.

## Build it

```
powershell -File scripts\build_persona.ps1 `
    -PersonaNames "her nickname","her other handle"
```

That produces `data/corpus/timeline.jsonl` - every message from every platform
merged into one chronological order - and then the training file.

## Sources get merged into one timeline

Message order is restored globally, not per chat, because the same conversation
continues across platforms. Sorting is deterministic on
`(timestamp, source, original_id)` so re-running produces byte-identical output.
