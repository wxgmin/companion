"""End-to-end chain test: orchestrator -> Hermes -> Command Code.

Verifies the full delegation path actually produces a file in the project
directory, rather than only checking that the subprocess exited zero. Earlier
runs reported success while writing to the home directory, so this checks
ground truth on disk.

    orchestrator\\.venv\\Scripts\\python.exe orchestrator\\test_chain.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import router as router_mod  # noqa: E402

TARGET = config.PROJECT / "chain_check.txt"
CODE = config.PROJECT / "chain_sum.py"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} {detail}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    print("=" * 70)
    print("  delegation chain: orchestrator -> hermes -> command code")
    print("=" * 70)

    TARGET.unlink(missing_ok=True)
    CODE.unlink(missing_ok=True)

    # 1. plain task through Hermes
    print("\n1. task routed to Hermes")
    t0 = time.perf_counter()
    out = router_mod.run_hermes(
        f"Create a file at {TARGET} containing exactly: chain ok", timeout=420
    )
    result, ok_flag = out
    print(f"     {time.perf_counter() - t0:.1f}s, exit_ok={ok_flag}")
    check("file created at the right path", TARGET.exists(), str(TARGET))
    if TARGET.exists():
        check("content correct", TARGET.read_text().strip() == "chain ok",
              repr(TARGET.read_text().strip()))

    # 2. coding task that should reach Command Code via the skill
    print("\n2. coding task routed through the command-code skill")
    t0 = time.perf_counter()
    result, ok_flag = router_mod.run_hermes(
        f"Use the command-code skill to delegate to cmdc. Create {CODE} "
        "containing a function add(a, b) that returns a + b. "
        "Do not write it yourself - delegate it.",
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    mentioned = "cmdc" in result.lower() or "command code" in result.lower()
    print(f"     {elapsed:.1f}s, exit_ok={ok_flag}")
    check("cmdc was used", mentioned, "found in report" if mentioned else "not mentioned")
    check("code file created in project", CODE.exists(), str(CODE))
    if CODE.exists():
        src = CODE.read_text()
        try:
            ns: dict = {}
            exec(compile(src, str(CODE), "exec"), ns)
            check("add() works", ns["add"](2, 3) == 5, f"add(2,3)={ns['add'](2,3)}")
        except Exception as exc:  # noqa: BLE001
            check("add() works", False, f"{type(exc).__name__}: {exc}")

    # 3. nothing should have leaked to the home directory this time
    print("\n3. working directory containment")
    stray = Path.home() / "chain_check.txt"
    check("nothing written to home dir", not stray.exists(), str(stray))

    print()
    print("=" * 70)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  CHAIN OK")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
