"""Boot-time checks so the stack fails loudly instead of quietly degrading.

The failure this exists to prevent: the LLM's context overflows VRAM, Ollama
spills layers onto the CPU without erroring, and the only symptom is that
everything is slow and tool calls stop parsing. Checking up front turns that
into a clear message.
"""

from __future__ import annotations

import subprocess
import sys

import config


def gpu_memory() -> tuple[int, int, int]:
    """(used_mib, free_mib, total_mib)"""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    used, free, total = (int(x.strip()) for x in out.split(","))
    return used, free, total


def report() -> dict:
    import llm
    import tts

    results: dict = {"checks": [], "ok": True}

    def add(name: str, ok: bool, detail: str, fatal: bool = True) -> None:
        results["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok and fatal:
            results["ok"] = False

    # --- GPU headroom -----------------------------------------------------
    # Free VRAM moves constantly while the desktop is in use, so this is
    # advisory: a low number is worth knowing about but is not proof of a
    # problem. The authoritative check is the processor split below, which says
    # whether any model layers were pushed onto the CPU.
    try:
        used, free, total = gpu_memory()
        add(
            "gpu",
            True,
            f"{used}/{total} MiB used, {free} MiB free "
            f"(soft floor {config.MIN_FREE_VRAM_MIB})",
            fatal=False,
        )
        results["vram"] = {"used": used, "free": free, "total": total}
    except Exception as exc:  # noqa: BLE001
        add("gpu", False, f"nvidia-smi failed: {exc}")

    # --- voice ------------------------------------------------------------
    h = tts.healthy()
    add(
        "chatterbox",
        bool(h.get("ok")),
        (
            f"ready, exaggeration={h.get('exaggeration')}"
            if h.get("ok")
            else f"not reachable at {config.TTS_URL}: {h.get('error')}"
        ),
    )

    # --- brain ------------------------------------------------------------
    try:
        loaded = llm.resident()
        names = [m.get("name", "?") for m in loaded]
        if config.PERSONA_MODEL in names:
            m = next(x for x in loaded if x.get("name") == config.PERSONA_MODEL)
            ctx = m.get("context_length") or m.get("context")
            size = m.get("size") or 0
            size_vram = m.get("size_vram") or 0
            size_gb = size / 1e9

            # The authoritative check. Ollama keeps layers on the CPU without
            # erroring; size_vram < size is the only reliable signal of it, and
            # the symptom is otherwise just "everything got slow".
            if size and size_vram:
                ratio = size_vram / size
                on_gpu = ratio >= 0.999
                split = "100% GPU" if on_gpu else f"{ratio * 100:.0f}% on GPU"
            else:
                on_gpu, split = True, "split unknown"

            ctx_ok = ctx is None or int(ctx) >= config.PERSONA_NUM_CTX
            detail = f"{size_gb:.1f} GB, ctx={ctx}, {split}"
            if not on_gpu:
                detail += "  <- layers on CPU; lower PERSONA_NUM_CTX"
            elif not ctx_ok:
                detail += f"  <- below configured {config.PERSONA_NUM_CTX}"
            add("ollama-model", on_gpu and ctx_ok, detail)
            results["model"] = {"ctx": ctx, "size": size, "size_vram": size_vram}
        else:
            add(
                "ollama-model",
                True,
                f"not loaded yet (loads on first turn at num_ctx={config.PERSONA_NUM_CTX})",
                fatal=False,
            )
    except Exception as exc:  # noqa: BLE001
        add("ollama-model", False, f"ollama unreachable: {exc}")

    # --- the footgun ------------------------------------------------------
    # A keep_alive of -1 pins the model forever and starves every other
    # service; that is what broke the last attempt.
    add(
        "keep-alive",
        isinstance(config.KEEP_ALIVE, str) or config.KEEP_ALIVE >= 0,
        f"keep_alive={config.KEEP_ALIVE!r} (must not pin VRAM forever)",
    )

    return results


def main() -> int:
    r = report()
    width = max(len(c["name"]) for c in r["checks"])
    print("preflight")
    for c in r["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']:<{width}}  {c['detail']}")
    print("  => " + ("ALL CHECKS PASS" if r["ok"] else "BLOCKED"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
