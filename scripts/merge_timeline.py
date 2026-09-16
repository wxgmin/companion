"""Merge every parsed conversation into ONE globally chronological timeline.

Why this exists
---------------
The chats overlap in time: the WhatsApp export and both Instagram accounts each
span the same months, and a single evening can appear in all three. WeClone
concatenates CSVs file-by-file and then pairs question/answer across the whole
list, so shipping one CSV per conversation means the pairing sees a made-up
order (all of WhatsApp, then all of account 1, then all of account 2).

Sorting everything into a single timeline fixes the ordering, and the matching
patch in qa_generator.match_qa resets its state machine whenever `room_name`
changes, so conversations never pair across each other.

Sort key is (CreateTime, room_name, original id) so that:
  * the timeline is chronological,
  * messages sharing a timestamp stay grouped with their own conversation,
  * the result is deterministic and stable across runs.

Usage:
    python merge_timeline.py --out OUT.csv IN1.csv IN2.csv [IN3.csv ...]
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

FIELDS = [
    "id",
    "MsgSvrID",
    "type_name",
    "is_sender",
    "talker",
    "room_name",
    "msg",
    "src",
    "CreateTime",
    "is_forward",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--preview", type=int, default=25,
                    help="how many timeline rows to preview showing interleaving")
    args = ap.parse_args()

    rows = []
    per_source = {}

    for i, path in enumerate(args.inputs):
        p = Path(path)
        if not p.exists():
            print(f"missing: {p}")
            return 2
        with open(p, encoding="utf-8", newline="") as f:
            src_rows = list(csv.DictReader(f))
        for r in src_rows:
            r["_orig_id"] = int(r["id"]) if r["id"].isdigit() else 0
            r["_src_order"] = i
        rows.extend(src_rows)
        per_source[p.stem] = len(src_rows)

    print("=== inputs ===")
    for k, v in per_source.items():
        print(f"  {v:7d}  {k}")
    print(f"  {len(rows):7d}  TOTAL before merge")

    # --- merge into one chronological timeline ---
    rows.sort(key=lambda r: (r["CreateTime"], r["room_name"], r["_src_order"], r["_orig_id"]))

    # reassign sequential ids in timeline order
    for n, r in enumerate(rows, 1):
        r["id"] = n
        r["MsgSvrID"] = f"tl-{n}"

    # --- report ---
    rooms = Counter(r["room_name"] for r in rows)
    print(f"\n=== conversations merged ===")
    for room, n in rooms.most_common():
        print(f"  {n:7d}  {room}")

    switches = sum(
        1 for a, b in zip(rows, rows[1:]) if a["room_name"] != b["room_name"]
    )
    times = sorted(r["CreateTime"] for r in rows)
    print(f"\n=== timeline ===")
    print(f"  rows            : {len(rows)}")
    print(f"  date range      : {times[0]}  ->  {times[-1]}")
    print(f"  room switches   : {switches}  (proves the conversations genuinely interleave)")

    # duplicate-time collisions across rooms (the dangerous case)
    from collections import defaultdict

    by_time = defaultdict(set)
    for r in rows:
        by_time[r["CreateTime"]].add(r["room_name"])
    collisions = {t: rs for t, rs in by_time.items() if len(rs) > 1}
    print(f"  timestamps shared across conversations: {len(collisions)}")

    print(f"\n=== timeline preview (first {args.preview}) ===")
    for r in rows[: args.preview]:
        who = "HER " if r["is_sender"] == "1" else "him "
        txt = r["msg"].replace("\n", " / ")[:52]
        print(f"  {r['CreateTime']}  {r['room_name'][:18]:18s} {who} {txt}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
