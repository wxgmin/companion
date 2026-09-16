# Her voice goes here

This directory holds the voice reference used for cloning. **Nothing in here is
committed to git** - your reference audio is personal and is not redistributed
with this project.

## The one file that matters

```
data/voice/reference.wav
```

That is the clip the TTS engine clones from. Everything else in this folder is
scratch output.

## Rules for a good reference

The engine conditions on this clip for every single thing it says, so its
quality sets the ceiling for the whole voice.

| | |
|---|---|
| **Length** | 3-10 seconds. Hard limit. Too short and the timbre is unstable; too long and it starts blending in room tone. |
| **Speakers** | Exactly one. A clip with a second voice teaches the model to drift between them. |
| **Music** | None. Background music bleeds into every generated word. |
| **Noise** | Quiet room. No fans, no traffic, no keyboard. |
| **Content** | Natural continuous speech. Avoid clipped one-word replies. |
| **Format** | 16-44.1 kHz mono WAV. The converter accepts mp4/m4a/opus and resamples. |

## Turning a voice note into a reference

Phone recordings are usually mp4/m4a with a bad sample rate and a loudness
problem. This converts, downmixes, resamples, and normalises in one step:

```powershell
ffmpeg -i "her_voice_note.mp4" `
    -ac 1 -ar 24000 -af "loudnorm=I=-16:LRA=11:TP=-1.5" `
    data\voice\reference.wav
```

Then **always check it before trusting it**:

```powershell
orchestrator\.venv\Scripts\python.exe scripts\check_ref_audio.py data\voice\reference.wav
```

That validates the length gate, the format, and prints a transcript so you can
confirm the clip actually contains what you think it does.

## If there is music behind her

Separate it first, then re-check. Whole clips that pass the length check can
still be unusable because the music is baked in:

```powershell
# isolate the vocal, then re-run the conversion above on the result
```

## Where to compare candidates

Different clips produce noticeably different voices. Drop candidates in here
and synthesise the same sentence from each before choosing - the difference is
much easier to hear side by side than in isolation:

```
data/voice/candidates/
    clipped_7s.wav
    full_12s.wav
    dereverbed.wav
```

Keep the winner as `reference.wav`. The `--reference` flag on the launcher
points at any other file without moving it:

```powershell
companion.cmd --reference data\voice\candidates\dereverbed.wav
```
