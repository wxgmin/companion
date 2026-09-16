---
name: command-code
description: "Delegate coding work to the Command Code agent (cmdc) headlessly. Use for multi-file code changes, refactors, debugging sessions, and anything that benefits from a coding-specialised agent with its own taste memory."
version: 1.0.0
author: Companion
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [coding, delegation, agent, refactor, command-code]
    related_skills: [software-development]
prerequisites:
  commands: [cmdc]
---

# Command Code

`cmdc` is a coding agent that runs on its own model and keeps a persistent
memory of how you like code written. Hand it work that is primarily *coding*
rather than a one-off shell command.

## When to use this instead of doing it yourself

Use it when the task is:

- A multi-file change that has to stay internally consistent
- A refactor, migration, or "make this match the rest of the codebase"
- Debugging where the fix is not yet obvious
- Anything you would otherwise spend many turns iterating on

Do it yourself when it is:

- A single command, a file read, or a quick lookup
- Something you already know exactly how to do in one or two steps

Delegating a trivial task costs more than it saves, because the agent has to
re-orient on the repository before it can start.

## Non-interactive invocation

Always use `-p` for headless runs. Without it the process opens an interactive
session and hangs forever.

**In print mode the file and shell tools are withheld by default.** A run will
report that it cannot write anything unless you explicitly re-enable them - this
is the single most common way a delegated task silently does nothing, because
the agent exits `0` after explaining that it had no tools:

```
Tool "write_file" requires permissions. Use --yolo (or --dangerously-skip-permissions)
to enable file writes and shell commands in print mode.
```

Pass either `--yolo`, or the narrower `--tools-all` / `--tools-enable` if you
want to keep some guardrails:

```bash
cmdc -p "Add a --verbose flag to the CLI and thread it through the logger" \
  --output-format text \
  --max-turns 40 \
  --yolo
```

Key flags:

| Flag | Why |
|---|---|
| `-p "query"` | Non-interactive; required when called from a script |
| `--output-format text` | Plain final answer. Use `json` for an NDJSON event stream |
| `--max-turns N` | Hard cap. Exits 8 if it hits the cap, so check the exit code |
| `--yolo` | **Required for writes in print mode.** Bypasses every permission check |
| `--tools-all` | Print mode: enable every tool, including withheld ones, keeping prompting |
| `--tools-enable <names>` | Print mode: enable specific withheld tools, comma-separated |
| `--auto-accept` | Only affects prompts in an *interactive* session; not enough for `-p` |
| `-c` / `--continue` | Continue the previous conversation instead of starting fresh |
| `--add-dir <dir>` | Give it extra workspace context |
| `-w` / `--worktree` | Run in an isolated git worktree, leaving the working tree untouched |

Prefer `--tools-all` over `--yolo` when the work is not disposable: it still
makes the agent ask before anything irreversible, where `--yolo` does not.

## Working directory matters

Run it from inside the repository you want changed. The agent orients on the
current directory, so invoking it from a home directory will either fail to
find the project or pick up the wrong one:

```bash
cd /path/to/project
cmdc -p "task"
```

## Exit codes

- `0` - completed
- `8` - hit `--max-turns` without finishing. The work may be partial; re-run with
  a higher cap or `-c` to continue, and say plainly that it did not finish.

Do not report success on exit 8.

## Reporting back

The output is written for a developer, not for speech. Summarise what changed in
one or two sentences before relaying it, and give the concrete evidence: files
touched, commands run, tests passing. If the exit code was 8 or the output is
ambiguous, say the work is incomplete rather than assuming it landed.

## Long runs

A real refactor can take minutes. Raise the timeout rather than killing it, and
prefer `-c` to resume over restarting from scratch:

```bash
cmdc -p "continue: finish the remaining call sites" -c --output-format text --max-turns 60
```
