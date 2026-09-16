"""Parse Instagram "Download your information" HTML message exports.

Source structure (one div per message):

    <div class="pam _3-95 _2ph- _a6-g uiBoxWhite noborder">
      <h2 class="_3-95 _2pim _a6-h _a6-i">SENDER</h2>
      <div class="_3-95 _a6-p">
        <div>
          <div></div>
          <div>Ya in the park ?</div>        <- the message
          <div></div>
        </div>
      </div>
      <div class="_3-94 _a6-o">Jun 11, 2026 1:38 pm</div>
    </div>

Two traps:
  * Messages are in REVERSE chronological order (newest first). We sort by the
    parsed timestamp rather than trusting document order.
  * Attachments nest extra divs holding the shared post's caption, hashtags and
    links. Those are not conversation and must not become training targets, so a
    message is classified from its FIRST non-empty inner div only.

ROLE INVERSION (same as parse_whatsapp.py): the persona being cloned gets
is_sender=1, and the user gets is_sender=0.

Usage:
    python parse_instagram.py --survey FILE...
    python parse_instagram.py --out OUT.csv --room NAME --me "Your Name" FILE...
"""

import argparse
import csv
import re
import sys
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

TS_RX = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{4}) (\d{1,2}):(\d{2}) (am|pm)$")

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Markers Instagram writes in place of real text.
MEDIA_PATTERNS = [
    re.compile(r"^you sent an attachment\.?$", re.I),
    re.compile(r"^.+ sent an attachment\.?$", re.I),
    re.compile(r"^you sent a photo\.?$", re.I),
    re.compile(r"^.+ sent a photo\.?$", re.I),
    re.compile(r"^you sent a video\.?$", re.I),
    re.compile(r"^.+ sent a video\.?$", re.I),
    re.compile(r"^you sent a voice message\.?$", re.I),
    re.compile(r"^.+ sent a voice message\.?$", re.I),
    re.compile(r"^click for video:?$", re.I),
    re.compile(r"^click for photo:?$", re.I),
    re.compile(r"^click to view.*$", re.I),
    re.compile(r"^\d+ photos?$", re.I),
    re.compile(r"^\d+ videos?$", re.I),
]

# Senders that are never the persona even though they are "not you" -- e.g. the
# Meta AI bot, which would otherwise be cloned as her.
BLOCKED_SENDERS = {"meta ai", "instagram", "facebook"}

# Instagram records call events as ordinary message divs ("Video chat ended",
# "You started a video chat", "Duration: 24 seconds"). They are system noise:
# if left in, they become conversational turns and can even be learned as the
# persona's replies. Matched per-LINE, because WeClone merges consecutive
# messages into one turn and can bury them mid-turn.
SYSTEM_EVENT = re.compile(
    r"^\s*(?:"
    r"(?:you |[\w .'\-]{1,30} )?(?:started|ended|missed) (?:a )?(?:video|audio) (?:chat|call)"
    r"|(?:video|audio) (?:chat|call) (?:ended|started|missed)"
    r"|duration:\s*\d+\s*(?:seconds?|minutes?|hours?)"
    r"|you missed (?:a )?(?:video|audio) (?:chat|call)"
    r"|missed (?:video|audio) (?:chat|call)"
    r"|(?:audio|video) call"
    r")\s*\.?$",
    re.I,
)

# System/event noise that must never become a training target.
SKIP_PATTERNS = [
    re.compile(r"^liked a message\.?$", re.I),
    re.compile(r"^.+ liked a message\.?$", re.I),
    re.compile(r"^reacted .* to your message\.?$", re.I),
    re.compile(r"^.+ reacted .* to your message\.?$", re.I),
    re.compile(r"^you reacted .* to .* message\.?$", re.I),
    re.compile(r"^you unsent a message\.?$", re.I),
    re.compile(r"^.+ unsent a message\.?$", re.I),
    re.compile(r"^this message was unsent\.?$", re.I),
    re.compile(r"^you changed the theme\.?$", re.I),
    re.compile(r"^.+ changed the theme\.?$", re.I),
    re.compile(r"^you started an audio call.*$", re.I),
    re.compile(r"^you started a video call.*$", re.I),
]

BARE_URL = re.compile(r"^https?://\S+$", re.I)


def parse_ts(s: str):
    m = TS_RX.match(s.strip())
    if not m:
        return None
    mon, day, year, hour, minute, mer = m.groups()
    h = int(hour) % 12
    if mer == "pm":
        h += 12
    try:
        return datetime(int(year), MONTHS[mon], int(day), h, int(minute))
    except Exception:  # noqa: BLE001
        return None


def inner_texts(content_div):
    """First-level non-empty text children of the message body."""
    body = content_div.find("div")
    if body is None:
        return []
    out = []
    for child in body.find_all("div", recursive=False):
        t = child.get_text("\n", strip=True)
        if t:
            out.append(t)
    return out


def classify(texts):
    """-> ('text'|'图片'|'粘贴的文本'|None, msg)"""
    if not texts:
        # A message box with no inner text is an attachment (photo / video /
        # reel). Keep it as a media row so it still acts as a conversation
        # boundary instead of being silently dropped.
        return "图片", ""

    # Flatten to lines and strip call/system events wherever they appear --
    # WeClone merges consecutive messages, so an event can end up mid-turn.
    lines = []
    for t in texts:
        for line in t.split("\n"):
            line = line.strip()
            if not line:
                continue
            if SYSTEM_EVENT.match(line):
                continue
            lines.append(line)

    if not lines:
        # Nothing but system events. Keep it as a boundary: a call genuinely
        # interrupts the conversation and the pairing logic should see that.
        return "图片", ""

    first = lines[0]
    for p in SKIP_PATTERNS:
        if p.match(first):
            return None, ""
    for p in MEDIA_PATTERNS:
        if p.match(first):
            return "图片", first
    if BARE_URL.match(first):
        return "粘贴的文本", first
    # Normal message. Join multiple real lines, but never pull in the deep
    # nested attachment block (that lives 2+ levels down).
    return "text", "\n".join(lines)


def load_messages(path: Path):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    out = []
    for box in soup.find_all("div", class_=lambda c: c and "uiBoxWhite" in c):
        h2 = box.find("h2")
        if not h2:
            continue
        sender = h2.get_text(" ", strip=True)
        content = box.find("div", class_=lambda c: c and "_a6-p" in c)
        ts_div = box.find("div", class_=lambda c: c and "_a6-o" in c)
        ts = parse_ts(ts_div.get_text(strip=True)) if ts_div else None
        texts = inner_texts(content) if content else []
        out.append({"sender": sender, "ts": ts, "texts": texts})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--room", default=None)
    ap.add_argument("--me", action="append", default=[],
                    help="sender name(s) that are YOU; repeatable")
    ap.add_argument("--persona", action="append", default=[],
                    help="sender name(s) that are HER; repeatable. Anything not "
                         "listed as --me/--persona is skipped (and reported).")
    args = ap.parse_args()

    if not args.survey and not args.out:
        print("need either --survey or --out")
        return 2

    all_msgs = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"missing: {p}")
            return 2
        msgs = load_messages(p)
        np_ts = sum(1 for m in msgs if m["ts"])
        senders = {}
        for m in msgs:
            senders[m["sender"]] = senders.get(m["sender"], 0) + 1
        print(f"\n=== {p.parent.name} / {p.name} ===")
        print(f"  message divs : {len(msgs)}")
        print(f"  parsed times : {np_ts}  (unparsed: {len(msgs) - np_ts})")
        if msgs:
            ordered = [m for m in msgs if m["ts"]]
            if ordered:
                print(f"  doc order    : {ordered[0]['ts']}  ->  {ordered[-1]['ts']}")
            print(f"  senders:")
            for s, n in sorted(senders.items(), key=lambda kv: -kv[1]):
                who = "YOU" if s in args.me else ("?" if not args.me else "PERSONA")
                print(f"    {n:6d}  {s!r}  [{who}]")
        all_msgs.extend(msgs)

    print(f"\nTOTAL message divs across all files: {len(all_msgs)}")

    if args.survey:
        print("\n=== distinct first-line patterns (top 25) ===")
        pats = {}
        for m in all_msgs:
            first = (m["texts"][0].strip() if m["texts"] else "<NO TEXT>")
            kind, _ = classify(m["texts"])
            key = first[:70]
            pats[key] = pats.get(key, 0) + 1
        for k, n in sorted(pats.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {n:6d}  {k!r}")
        return 0

    # --- emit CSV ---
    me_names = {s.lower() for s in args.me}
    persona_names = {s.lower() for s in args.persona}

    rows = []
    stats = {"text": 0, "media": 0, "url": 0, "skipped": 0, "notime": 0,
             "blocked_sender": 0, "unknown_sender": 0}
    unknown_seen = {}
    blocked_seen = {}

    for m in all_msgs:
        s = m["sender"].strip()
        sl = s.lower()

        if sl in BLOCKED_SENDERS:
            stats["blocked_sender"] += 1
            blocked_seen[s] = blocked_seen.get(s, 0) + 1
            continue
        if sl in me_names:
            is_sender = 0
        elif persona_names:
            if sl in persona_names:
                is_sender = 1
            else:
                # Explicit persona list supplied: never guess.
                stats["unknown_sender"] += 1
                unknown_seen[s] = unknown_seen.get(s, 0) + 1
                continue
        else:
            is_sender = 1

        if m["ts"] is None:
            stats["notime"] += 1
            continue
        type_name, msg = classify(m["texts"])
        if type_name is None:
            stats["skipped"] += 1
            continue
        if type_name == "text":
            stats["text"] += 1
        elif type_name == "粘贴的文本":
            stats["url"] += 1
        else:
            stats["media"] += 1
        rows.append(
            {
                "id": 0,
                "MsgSvrID": "",
                "type_name": type_name,
                "is_sender": is_sender,
                "talker": m["sender"],
                "room_name": args.room or Path(args.files[0]).parent.name,
                "msg": msg if msg.strip() else type_name,
                "src": "",
                "CreateTime": m["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                "is_forward": "",
            }
        )

    # Sort chronologically: document order is reverse-chronological.
    rows.sort(key=lambda r: r["CreateTime"])

    # Instagram splits a conversation across message_1/message_2 by time slice,
    # and those slices can share a boundary message. drop exact duplicates.
    seen = set()
    deduped = []
    dropped = 0
    for r in rows:
        key = (r["is_sender"], r["CreateTime"], r["msg"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(r)
    rows = deduped
    if dropped:
        print(f"\n  deduped {dropped} boundary-duplicate rows")

    for i, r in enumerate(rows, 1):
        r["id"] = i
        r["MsgSvrID"] = f"ig-{i}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    hers = sum(1 for r in rows if r["is_sender"] == 1)
    print(f"\n=== OUTPUT ===")
    print(f"  text rows     : {stats['text']}")
    print(f"  media rows    : {stats['media']}")
    print(f"  url rows      : {stats['url']}")
    print(f"  skipped(sys)  : {stats['skipped']}")
    print(f"  no timestamp  : {stats['notime']}")
    print(f"  blocked sender: {stats['blocked_sender']}  {dict(blocked_seen)}")
    print(f"  unknown sender: {stats['unknown_sender']}")
    if unknown_seen:
        print(f"    (not you/persona, dropped): {dict(unknown_seen)}")
    print(f"  wrote         : {len(rows)} rows -> {out}")
    print(f"    persona     : {hers}")
    print(f"    you         : {len(rows) - hers}")
    if rows:
        print(f"  date range    : {rows[0]['CreateTime']}  ->  {rows[-1]['CreateTime']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
