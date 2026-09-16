"""One-command launcher for the whole companion stack.

Starts, in order, only what is not already running:

  1. Ollama, and pulls the persona model if it is missing
  2. Chatterbox TTS with her cloned reference voice
  3. Checks Hermes (the agent that does real work) is on PATH
  4. Runs preflight so a VRAM or wiring problem surfaces before you talk
  5. Starts the orchestrator server and, optionally, the voice loop

Everything is idempotent: running it twice reuses what is already up instead of
starting duplicates or fighting over ports.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import config

PROJECT = config.PROJECT
SCRIPTS = PROJECT / "scripts"
CBCLI = PROJECT / "chatterbox"
CB_PY = CBCLI / ".venv" / "Scripts" / "python.exe"
CB_SERVER = SCRIPTS / "chatterbox_server.py"
REFERENCE = PROJECT / "data" / "voice" / "reference.wav"

_started: list[tuple[str, subprocess.Popen]] = []


# ------------------------------------------------------------------ pretty
def say(msg: str) -> None:
    print(msg, flush=True)


def step(n: int, total: int, msg: str) -> None:
    say(f"\n[{n}/{total}] {msg}")


def ok(msg: str) -> None:
    say(f"      OK   {msg}")


def warn(msg: str) -> None:
    say(f"      WARN {msg}")


def fail(msg: str) -> None:
    say(f"      FAIL {msg}")


# ------------------------------------------------------------------ probes
def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        import httpx

        return httpx.get(url, timeout=timeout).status_code < 500
    except Exception:  # noqa: BLE001
        return False


def wait_for(predicate, label: str, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(2)
    warn(f"{label} did not come up within {timeout:.0f}s")
    return False


def spawn(name: str, args: list[str], cwd: Path | None = None, env: dict | None = None):
    merged = {**os.environ, **(env or {})}
    merged.setdefault("PYTHONUTF8", "1")
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(args, cwd=str(cwd or PROJECT), env=merged)
    _started.append((name, proc))
    return proc


# ------------------------------------------------------------------- steps
def ensure_ollama() -> bool:
    if http_ok(f"{config.OLLAMA_URL}/api/tags"):
        ok(f"ollama already running at {config.OLLAMA_URL}")
    else:
        if not shutil.which("ollama"):
            fail("ollama not found on PATH - install it first")
            return False
        say("      starting ollama ...")
        spawn("ollama", ["ollama", "serve"])
        if not wait_for(lambda: http_ok(f"{config.OLLAMA_URL}/api/tags"), "ollama"):
            return False
        ok("ollama started")

    try:
        import httpx

        tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=15).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
    except Exception as exc:  # noqa: BLE001
        fail(f"could not list models: {exc}")
        return False

    if config.PERSONA_MODEL in names:
        ok(f"persona model present: {config.PERSONA_MODEL}")
        return True

    warn(f"model {config.PERSONA_MODEL} not installed - pulling (this is large)")
    result = subprocess.run(["ollama", "pull", config.PERSONA_MODEL], cwd=str(PROJECT))
    if result.returncode != 0:
        fail("pull failed")
        return False
    ok("model pulled")
    return True


def ensure_tts(exaggeration: float, ref: Path) -> bool:
    if http_ok(f"{config.TTS_URL}/health"):
        ok(f"chatterbox already serving at {config.TTS_URL}")
        return True

    if not CB_PY.exists():
        fail(f"chatterbox venv missing at {CB_PY}")
        say("         run: scripts\\setup_chatterbox.ps1")
        return False
    if not CB_SERVER.exists():
        fail(f"server script missing at {CB_SERVER}")
        return False
    if not ref.exists():
        fail(f"reference voice missing at {ref}")
        say("         put her 3-10 second clip there - see VOICE.md")
        return False

    say(f"      starting chatterbox (encoding reference voice) ...")
    log = PROJECT / "logs" / "chatterbox.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        proc = subprocess.Popen(
            [
                str(CB_PY),
                "-u",
                str(CB_SERVER),
                "--ref",
                str(ref),
                "--exaggeration",
                str(exaggeration),
                "--port",
                str(config.TTS_URL.rsplit(":", 1)[-1]),
            ],
            cwd=str(CBCLI),
            env={**os.environ, "PYTHONUTF8": "1"},
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    _started.append(("chatterbox", proc))

    if not wait_for(lambda: http_ok(f"{config.TTS_URL}/health"), "chatterbox", 240):
        fail(f"see {log}")
        return False
    ok(f"chatterbox ready (voice={config.TTS_VOICE}, exaggeration={exaggeration})")
    return True


def check_hermes() -> bool:
    exe = shutil.which("hermes")
    if not exe:
        warn("hermes not on PATH - she can still chat but cannot run tasks")
        return False
    ok(f"hermes agent found: {exe}")
    return True


def run_preflight() -> bool:
    import preflight

    r = preflight.report()
    width = max(len(c["name"]) for c in r["checks"])
    for c in r["checks"]:
        mark = "OK  " if c["ok"] else "FAIL"
        say(f"      [{mark}] {c['name']:<{width}}  {c['detail']}")
    return bool(r["ok"])


def start_server() -> bool:
    if port_open(config.ORCHESTRATOR_PORT):
        ok(f"server already listening on {config.ORCHESTRATOR_PORT}")
        return True
    log = PROJECT / "logs" / "orchestrator.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        proc = subprocess.Popen(
            [str(Path(sys.executable)), "-u", "server.py"],
            cwd=str(PROJECT / "orchestrator"),
            env={**os.environ, "PYTHONUTF8": "1"},
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    _started.append(("orchestrator", proc))
    if not wait_for(lambda: port_open(config.ORCHESTRATOR_PORT), "orchestrator", 90):
        fail(f"see {log}")
        return False
    ok(f"server ready at http://{config.ORCHESTRATOR_HOST}:{config.ORCHESTRATOR_PORT}")
    return True


def shutdown(*_args) -> None:
    say("\nshutting down what this script started ...")
    for name, proc in reversed(_started):
        if proc.poll() is None:
            say(f"  stopping {name}")
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
    say("done")


def main() -> int:
    ap = argparse.ArgumentParser(description="Start the whole companion stack")
    ap.add_argument("--voice", action="store_true", help="open the mic loop after startup")
    ap.add_argument("--no-serve", action="store_true", help="skip the HTTP server")
    ap.add_argument("--exaggeration", type=float, default=config.TTS_EXAGGERATION)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--keep-running", action="store_true",
                    help="leave services up after the voice loop exits")
    args = ap.parse_args()

    say("=" * 62)
    say("  companion startup")
    say("=" * 62)

    total = 4 if args.no_serve else 5
    n = 0

    n += 1
    step(n, total, "language model")
    if not ensure_ollama():
        return 1

    n += 1
    step(n, total, "voice (chatterbox)")
    if not ensure_tts(args.exaggeration, args.reference):
        return 1

    n += 1
    step(n, total, "task agent")
    check_hermes()

    n += 1
    step(n, total, "preflight")
    if not run_preflight():
        fail("preflight did not pass - fix the above first")
        return 1

    if not args.no_serve:
        n += 1
        step(n, total, "orchestrator server")
        if not start_server():
            return 1

    say("\n" + "=" * 62)
    say("  READY")
    say(f"  chat API : http://{config.ORCHESTRATOR_HOST}:{config.ORCHESTRATOR_PORT}/docs")
    say(f"  ws       : ws://{config.ORCHESTRATOR_HOST}:{config.ORCHESTRATOR_PORT}/ws")
    say("=" * 62)

    if args.voice:
        say("\nstarting voice loop - just start talking\n")
        from voice_loop import VoiceLoop

        try:
            VoiceLoop().run()
        finally:
            if not args.keep_running:
                shutdown()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        shutdown()
        raise SystemExit(130)
