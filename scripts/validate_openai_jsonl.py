"""Validate an OpenAI chat fine-tuning JSONL before uploading (and paying).

Checks:
  * every line is valid JSON with a `messages` list
  * roles are only system/user/assistant, and the first non-system turn is user
  * consecutive same-role turns (OpenAI merges these, but it's a smell)
  * at least one assistant turn, and no empty contents
  * per-example size against the model's context ceiling
  * leftover artefacts that should never have survived: <begin_chat>, Chinese
    placeholder text, bare media markers, <image>

Usage:
    python validate_openai_jsonl.py FILE.jsonl [--model gpt-4.1-mini]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Approximate per-example training ceilings for chat fine-tuning.
MODEL_LIMITS = {
    "gpt-4.1-mini": 16384,
    "gpt-4.1-nano": 16384,
    "gpt-4.1": 16384,
    "gpt-4o-mini": 16384,
    "gpt-4o": 16384,
}

BAD_PATTERNS = {
    "begin_chat": re.compile(r"<begin_chat>|<end_chat>"),
    "chinese_placeholder": re.compile(r"[\u4e00-\u9fff]"),
    "image_placeholder": re.compile(r"<image>"),
    "media_marker_only": re.compile(r"^\s*(图片|视频|语音|粘贴的文本|未知)\s*$"),
    "bare_url": re.compile(r"^\s*https?://\S+\s*$", re.I),
}


def approx_tokens(text: str) -> int:
    # chars/4 is the usual rough English heuristic
    return max(1, len(text) // 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"not found: {p}")
        return 2

    limit = MODEL_LIMITS.get(args.model, 16384)
    rows = []
    errors = []

    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {ln}: invalid JSON ({e})")
            continue
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"line {ln}: missing/empty `messages`")
            continue
        rows.append((ln, msgs))

    print(f"file         : {p}")
    print(f"lines parsed : {len(rows)}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  {e}")

    role_counts = Counter()
    size_problems = []
    artefact_counts = Counter()
    no_assistant = 0
    empty_content = 0
    consecutive_same = 0
    first_not_user = 0
    sizes = []

    for ln, msgs in rows:
        total_chars = 0
        roles = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content") or ""
            role_counts[role] += 1
            roles.append(role)
            total_chars += len(content)
            if not isinstance(content, str) or not content.strip():
                empty_content += 1
            for name, rx in BAD_PATTERNS.items():
                if rx.search(content):
                    artefact_counts[name] += 1
        tok = approx_tokens("".join(m.get("content") or "" for m in msgs))
        sizes.append(tok)
        if tok > limit:
            size_problems.append((ln, tok))
        if "assistant" not in roles:
            no_assistant += 1
        non_sys = [r for r in roles if r != "system"]
        if non_sys and non_sys[0] != "user":
            first_not_user += 1
        for a, b in zip(roles, roles[1:]):
            if a == b:
                consecutive_same += 1

    print(f"\n=== roles ===")
    for r, n in role_counts.most_common():
        print(f"  {r:10s} {n}")

    print(f"\n=== structure ===")
    print(f"  examples without an assistant turn : {no_assistant}")
    print(f"  examples not starting with user    : {first_not_user}")
    print(f"  empty content turns                : {empty_content}")
    print(f"  consecutive same-role pairs        : {consecutive_same}")

    print(f"\n=== artefacts that should be zero ===")
    if artefact_counts:
        for k, v in artefact_counts.most_common():
            print(f"  {k:22s} {v}")
    else:
        print("  none")

    print(f"\n=== size (est tokens, model limit {limit}) ===")
    if sizes:
        sizes_sorted = sorted(sizes)
        print(f"  min {sizes_sorted[0]}  median {sizes_sorted[len(sizes_sorted)//2]}  max {sizes_sorted[-1]}")
        tot = sum(sizes)
        print(f"  total {tot:,} tokens  (~{tot*3/1e6:.2f}M over 3 epochs)")
    if size_problems:
        print(f"  OVER LIMIT: {len(size_problems)} examples")
        for ln, tok in size_problems[:10]:
            print(f"    line {ln}: ~{tok} tokens")
    else:
        print("  all examples within limit")

    ok = (
        not errors
        and not size_problems
        and not artefact_counts
        and no_assistant == 0
        and first_not_user == 0
        and empty_content == 0
    )
    print(f"\nVERDICT: {'READY TO UPLOAD' if ok else 'NEEDS ATTENTION'}")

    if args.show:
        print(f"\n=== {args.show} sample(s) ===")
        for ln, msgs in rows[: args.show]:
            print(f"\n--- line {ln} ---")
            for m in msgs:
                c = (m.get("content") or "").replace("\n", " ⏎ ")
                print(f"  {m.get('role'):9s}: {c[:150]}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
