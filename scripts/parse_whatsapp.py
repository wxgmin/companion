"""Parse a WhatsApp "Export chat" .txt into WeClone's CSV contract.

Output columns (must match WeClone's ChatMessage / telegram_parser to_csv):
    id, MsgSvrID, type_name, is_sender, talker, room_name, msg, src, CreateTime, is_forward

ROLE INVERSION -- the single most important detail here
-------------------------------------------------------
WeClone clones *you*: in its state machine `is_sender == 1` is the "own message"
that becomes the training response, and `is_sender == 0` is the other party's
message that becomes the instruction.

We want the model to reply AS HER. So:

    HER messages  -> is_sender = 1   (the persona being cloned)
    HIS messages  -> is_sender = 0   (the prompt / context)

Getting this backwards would train a model to impersonate him.

type_name
---------
WeClone's `chat` platform compares type_name against its zh_CN vocabulary
(GPT-SoVITS-style cut/skip lists), so:
    plain text            -> exactly "text"
    image / video / voice -> 图片 / 视频 / 语音   (act as conversation boundaries)
    bare URLs             -> 粘贴的文本 (so they never become training targets)

Observed source format (Android, English-locale with dotted meridiem):
    2024-12-20, 10:23 a.m. - Waiz: Omggg soo cool
    continuation lines have NO date prefix and belong to the message above
    system notices have no "Sender: " segment at all

Usage:
    python parse_whatsapp.py SRC.txt --out OUT.csv
"""

import argparse
import csv
import re
from datetime import datetime
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

# 2024-12-20, 10:23 a.m. - Sender: message
LINE_RX = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}), (?P<time>\d{1,2}:\d{2}\s[ap]\.m\.) - (?P<rest>.*)$"
)
# "a.m." / "p.m." -> AM/PM
TIME_RX = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})\s(?P<meridiem>[ap])\.m\.$")

EDIT_TAG = re.compile(r"\s*<This message was edited>\s*$")
MEDIA_OMITTED = re.compile(r"^<Media omitted>\s*$", re.I)
ATTACHED = re.compile(r"^(?P<name>[^\s]+\.(?P<ext>[A-Za-z0-9]{2,5})) \(file attached\)\s*$")
BARE_URL = re.compile(r"^https?://\S+$", re.I)

EXT_TYPE = {
    "jpg": "图片", "jpeg": "图片", "png": "图片", "webp": "图片", "gif": "图片",
    "mp4": "视频", "mov": "视频", "3gp": "视频", "mkv": "视频",
    "opus": "语音", "ogg": "语音", "m4a": "语音", "mp3": "语音", "aac": "语音",
}

SKIP_BODY = {
    "this message was deleted",
    "you deleted this message",
    "this message was deleted.",
}


def parse_dt(date_s: str, time_s: str) -> datetime:
    mt = TIME_RX.match(time_s)
    if not mt:
        raise ValueError(f"bad time: {time_s!r}")
    hour = int(mt.group("h")) % 12
    if mt.group("meridiem") == "p":
        hour += 12
    d = datetime.strptime(date_s, "%Y-%m-%d")
    return d.replace(hour=hour, minute=int(mt.group("m")))


def classify(body: str):
    """-> (type_name, msg, src)"""
    body = EDIT_TAG.sub("", body).strip()

    if MEDIA_OMITTED.match(body):
        return "图片", "", ""

    m = ATTACHED.match(body)
    if m:
        ext = m.group("ext").lower()
        return EXT_TYPE.get(ext, "未知"), "", m.group("name")

    if BARE_URL.match(body):
        return "粘贴的文本", body, ""

    return "text", body, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--me", default="Waiz", help="your sender name (is_sender=0)")
    ap.add_argument("--room", default=None, help="room_name; defaults to file stem")
    ap.add_argument("--parties", default=2, type=int,
                    help="expected sender count; only used for validation warnings")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"not found: {src}")
        return 2

    text = src.read_text(encoding="utf-8", errors="replace")
    raw_lines = text.split("\n")

    room = args.room or src.stem

    records = []
    senders = {}
    stats = {
        "message_lines": 0,
        "continuations": 0,
        "system_skipped": 0,
        "deleted_skipped": 0,
        "empty_as_media": 0,
        "unparsed": 0,
    }

    cur = None  # dict for the message being built

    def flush():
        if not cur:
            return
        body = cur["body"].strip()
        if body.lower() in SKIP_BODY:
            stats["deleted_skipped"] += 1
            return
        if not body:
            # A "Without media" export emits `Sender: ` with an EMPTY body for
            # attachments. These must be KEPT as media rows, not dropped:
            # WeClone treats non-text rows as conversation boundaries, so
            # silently dropping them lets it pair a message from before a photo
            # with a reply from after it -- producing mismatched QA pairs.
            stats["empty_as_media"] += 1
            type_name, msg, src_file = "图片", "", ""
        else:
            type_name, msg, src_file = classify(body)
        is_sender = 0 if cur["sender"] == args.me else 1
        records.append(
            {
                "id": len(records) + 1,
                "MsgSvrID": f"wa-{len(records) + 1}",
                "type_name": type_name,
                "is_sender": is_sender,
                "talker": cur["sender"],
                "room_name": room,
                "msg": (msg if msg.strip() else type_name) if type_name != "text" else msg,
                "src": src_file,
                "CreateTime": cur["dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "is_forward": "",
            }
        )

    for line in raw_lines:
        m = LINE_RX.match(line)
        if not m:
            if cur is not None:
                cur["body"] += "\n" + line
                stats["continuations"] += 1
            continue

        stats["message_lines"] += 1
        rest = m.group("rest")
        sep = rest.find(": ")
        head = rest[:sep] if sep != -1 else ""

        # A real message needs a "Sender: " prefix. System notices have none
        # (e.g. the end-to-end encryption banner), and are not too long.
        if sep == -1 or len(head) > 60 or "\n" in head:
            flush()
            cur = None
            stats["system_skipped"] += 1
            continue

        sender, body = rest[:sep], rest[sep + 2 :]
        senders[sender] = senders.get(sender, 0) + 1

        flush()
        try:
            dt = parse_dt(m.group("date"), m.group("time"))
        except Exception as exc:  # noqa: BLE001
            stats["unparsed"] += 1
            print(f"  ! timestamp parse failed ({exc}): {line[:80]}")
            cur = None
            continue
        cur = {"sender": sender, "dt": dt, "body": body}

    flush()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)

    hers = sum(1 for r in records if r["is_sender"] == 1)
    his = len(records) - hers

    print(f"\n=== parsed {src.name} ===")
    print(f"  raw lines        : {len(raw_lines)}")
    print(f"  message lines    : {stats['message_lines']}")
    print(f"  continuations    : {stats['continuations']} (appended to prior msg)")
    print(f"  system skipped   : {stats['system_skipped']}")
    print(f"  deleted skipped  : {stats['deleted_skipped']}")
    print(f"  blank->media     : {stats['empty_as_media']}  (blank-body attachments kept as boundaries)")
    print(f"  unparsed         : {stats['unparsed']}")
    print(f"\n  senders seen:")
    for s, n in sorted(senders.items(), key=lambda kv: -kv[1]):
        role = "YOU (is_sender=0)" if s == args.me else "PERSONA (is_sender=1)"
        print(f"    {n:6d}  {s!r}  -> {role}")
    print(f"\n  OUTPUT rows      : {len(records)}")
    print(f"    persona (her)  : {hers}")
    print(f"    you            : {his}")
    if len(senders) != args.parties:
        print(f"  ! expected {args.parties} senders, saw {len(senders)}")
    print(f"  written -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
