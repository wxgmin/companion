"""Decides whether a message is conversation or work, and drives Hermes for work.

Rather than pattern-matching keywords like "open" or "find", the persona model
is given a single ``delegate_task`` tool and chooses for herself. That keeps the
split between chatting and doing invisible, and it means she narrates her own
completed work in character instead of reading out a raw tool log.

The task itself is handed to the Hermes agent, which runs the tools and returns
a report; that report is fed back to the persona as a tool result so the spoken
reply stays conversational.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
import llm

SESSION_DIR = config.PROJECT / "data" / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

HERMES = "hermes"
TASK_TIMEOUT_S = float(os.getenv("COMPANION_TASK_TIMEOUT", "600"))

DELEGATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": (
            "Hand real work to the background agent: anything that touches the "
            "filesystem, runs commands, browses the web, writes or edits code, "
            "or needs more than one step. Do NOT use this for ordinary "
            "conversation, feelings, opinions, or questions you can already "
            "answer yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "A complete, self-contained instruction for the agent, "
                        "including every detail it needs without seeing this chat."
                    ),
                }
            },
            "required": ["task"],
        },
    },
}


@dataclass
class Outcome:
    reply: str
    task: str | None = None
    task_result: str | None = None
    tokens: int = 0
    seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tok_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


def run_hermes(task: str, timeout: float = TASK_TIMEOUT_S) -> tuple[str, bool]:
    """Run one non-interactive Hermes task. Returns (output, succeeded)."""
    # The task text comes from a model that may invent Unix paths; the agent
    # works on Windows, so the real facts are prepended rather than trusted.
    framed = (
        f"{config.ENVIRONMENT}\n\n"
        f"Task: {task}\n\n"
        "When you are done, state plainly whether it actually succeeded and "
        "give the concrete evidence (file path, command output). If it failed, "
        "say exactly what failed."
    )
    # Hermes resolves its shell cwd from terminal.cwd, which defaults to '.'
    # and lands in the home directory - NOT the directory this process was
    # launched from. Without TERMINAL_CWD a delegated task silently writes its
    # output into the wrong folder and still reports success. The flag --in
    # only scopes session lookup, it does not move the shell.
    env = {**os.environ, "TERMINAL_CWD": str(config.PROJECT)}
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [HERMES, "-z", framed, "--cli", "--no-restore-cwd"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(config.PROJECT),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"The task ran past {timeout:.0f}s and was stopped.", False
    except FileNotFoundError:
        return "The hermes command was not found on PATH.", False

    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and not out:
        out = (proc.stderr or "").strip()
    print(f"  [hermes] {elapsed:.1f}s exit={proc.returncode}", flush=True)
    return out or "(the agent returned nothing)", proc.returncode == 0


class Router:
    """Keeps short-term history and routes each turn to chat or to work."""

    def __init__(self, history_limit: int = 24, session: str | None = None) -> None:
        self.history: list[dict[str, Any]] = [llm.system_message()]
        self.history_limit = history_limit
        self.session_path = SESSION_DIR / (
            session or time.strftime("session-%Y%m%d-%H%M%S.jsonl")
        )

    # ---------------------------------------------------------------- log
    def _log(self, role: str, content: str) -> None:
        try:
            with self.session_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"ts": time.time(), "role": role, "content": content},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError:
            pass

    def _trim(self) -> None:
        """Keep the system prompt plus the last N turns."""
        if len(self.history) > self.history_limit + 1:
            self.history = [self.history[0]] + self.history[-self.history_limit :]

    # ------------------------------------------------------------ routing
    def respond(self, text: str) -> Outcome:
        self.history.append({"role": "user", "content": text})
        self._log("you", text)

        first = llm.chat(self.history, tools=[DELEGATE_TOOL])
        tokens = first.tokens
        seconds = first.seconds

        if not first.tool_calls:
            reply = first.text or "..."
            self.history.append({"role": "assistant", "content": reply})
            self._log("her", reply)
            self._trim()
            return Outcome(reply=reply, tokens=tokens, seconds=seconds)

        call = first.tool_calls[0]
        task = str(call.arguments.get("task") or "").strip()
        print(f"  [task] {task}", flush=True)

        result, ok = run_hermes(task)
        print(f"  [task] {'done' if ok else 'incomplete'}", flush=True)

        # Feed the report back so the spoken answer stays in character.
        self.history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    }
                ],
            }
        )
        self.history.append(
            {
                "role": "tool",
                "content": (
                    f"{result}\n\n"
                    "Now tell me what happened in your own voice, briefly and "
                    "casually. Do not read this out verbatim and do not mention "
                    "that a tool or agent ran. If the report does not clearly "
                    "show the work succeeded, say that it did not work instead "
                    "of claiming it did."
                ),
            }
        )

        second = llm.chat(self.history)
        reply = second.text or "done. it's handled."
        self.history.append({"role": "assistant", "content": reply})
        self._log("her", reply)
        self._trim()

        return Outcome(
            reply=reply,
            task=task,
            task_result=result,
            tokens=tokens + second.tokens,
            seconds=seconds + second.seconds,
        )

    def reset(self) -> None:
        self.history = [llm.system_message()]
