"""Verify the training-target direction: HIS messages must be context only.

The persona being cloned is HER. In WeClone's state machine:

    is_sender == 1  ->  "own message"  ->  becomes the ASSISTANT turn (trained on)
    is_sender == 0  ->  "other party"  ->  becomes the USER turn (context only)

Nothing belonging to "Waiz" may ever appear as an assistant turn.

Two independent checks:
  A. Source CSV: every is_sender=1 row must be HER, every is_sender=0 row must be
     HIM. Any mismatch is a parser bug and is reported loudly.
  B. Training JSONL: every assistant turn must be traceable to a message she
     actually sent, and must NOT be uniquely traceable to one of his.

Usage:
    python verify_role_inversion.py --csv TIMELINE.csv --jsonl train.jsonl \
        --her "name1" --her "name2" --him "Waiz"
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip()).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--her", action="append", required=True)
    ap.add_argument("--him", action="append", required=True)
    args = ap.parse_args()

    her_names = {n.lower() for n in args.her}
    him_names = {n.lower() for n in args.him}

    # ---------- A. source CSV ----------
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8", newline="")))
    wrong_her = []   # marked as the persona but sent by him
    wrong_him = []   # marked as context but sent by her
    for r in rows:
        who = (r["talker"] or "").lower()
        if r["is_sender"] == "1" and who in him_names:
            wrong_her.append(r)
        if r["is_sender"] == "0" and who in her_names:
            wrong_him.append(r)

    print("=" * 70)
    print("A. SOURCE CSV ROLE CHECK")
    print("=" * 70)
    print(f"  rows total                        : {len(rows)}")
    print(f"  is_sender=1 but sent by HIM       : {len(wrong_her)}")
    print(f"  is_sender=0 but sent by HER       : {len(wrong_him)}")

    if wrong_her:
        print("\n  !! HIM messages marked as the training target:")
        for r in wrong_her[:5]:
            print(f"     {r['CreateTime']}  {r['talker']}  {r['msg'][:60]}")
    if wrong_him:
        print("\n  !! HER messages marked as context only:")
        for r in wrong_him[:5]:
            print(f"     {r['CreateTime']}  {r['talker']}  {r['msg'][:60]}")
    if not wrong_her and not wrong_him:
        print("\n  OK: role assignment is consistent for every row.")

    # ---------- B. training JSONL ----------
    her_msgs = set()
    him_msgs = set()
    for r in rows:
        v = norm(r["msg"])
        if not v:
            continue
        if (r["talker"] or "").lower() in her_names:
            her_msgs.add(v)
        elif (r["talker"] or "").lower() in him_names:
            him_msgs.add(v)

    examples = []
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(json.loads(line))

    total_asst = 0
    traced_her = 0
    only_him = []
    untraced = []

    for ex in examples:
        for m in ex["messages"]:
            if m["role"] != "assistant":
                continue
            total_asst += 1
            content = m["content"]
            # try the whole turn, then each of its lines (turns are joins)
            candidates = [norm(content)] + [norm(x) for x in content.split("\n") if x.strip()]
            in_her = any(c in her_msgs for c in candidates if c)
            in_him = any(c in him_msgs for c in candidates if c)
            if in_her:
                traced_her += 1
            elif in_him:
                only_him.append(content)
            else:
                untraced.append(content)

    print()
    print("=" * 70)
    print("B. TRAINING JSONL TARGET CHECK")
    print("=" * 70)
    print(f"  examples                          : {len(examples)}")
    print(f"  assistant turns total             : {total_asst}")
    print(f"  traced to HER messages            : {traced_her}  "
          f"({100 * traced_her / max(total_asst, 1):.1f}%)")
    print(f"  traced ONLY to HIS messages       : {len(only_him)}  <-- must be 0")
    print(f"  untraced (merge/split artefacts)  : {len(untraced)}")

    if only_him:
        print("\n  !! These assistant turns came from HIM:")
        for c in only_him[:10]:
            print(f"     {c[:100]!r}")

    ok = not wrong_her and not wrong_him and not only_him
    print()
    print("VERDICT:", "ROLE DIRECTION CORRECT" if ok else "PROBLEM FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
