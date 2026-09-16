"""Generate a tiny synthetic WeClone CSV fixture to validate the pipeline.

Purpose: prove the parser -> CSV -> QA-pair -> SFT-dataset path works (and that
pandas 3.x doesn't break it) BEFORE we point it at the real chat logs.

CRITICAL ROLE INVERSION
-----------------------
WeClone is built to clone *you*. In its state machine
(weclone/data/qa_generator.py, WAITING_INSTRUCTION / WAITING_RESPONSE):

    is_sender == 0  ->  "Received message from other party"  -> becomes the INSTRUCTION
    is_sender == 1  ->  "Own message"                        -> becomes the RESPONSE (the target)

We want the model to reply AS HER. So in our CSVs:

    HER messages  -> is_sender = 1   (the persona being cloned)
    HIS messages  -> is_sender = 0   (the prompt / context)

type_name: platform "chat" compares against the zh_CN vocabulary
(GPT_SoVITS-style cut/skip lists), so plain text must be exactly "text".
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

OUT_DIR = Path(r"C:\Users\Waiz\ai-companion\WeClone\dataset\csv\test-thread")
OUT_CSV = OUT_DIR / "test-thread.csv"

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

HER = "Her"
HIM = "Me"
ROOM = "test-thread"

# (minutes_from_session_start, is_her, text)
SESSION_1 = [
    (0, False, "hey you up?"),
    (1, True, "yeah just got back"),
    (2, False, "how was the thing with your sister"),
    (3, True, "ugh dont even"),
    (4, True, "she started talking about the wedding again"),
    (5, False, "lol"),
    (6, False, "what did you say"),
    (7, True, "i just nodded and changed the subject"),
    (8, True, "im not doing a 200 person wedding"),
    (9, False, "fair enough"),
    (10, True, "anyway how was your day"),
    (11, False, "long. meetings all afternoon"),
    (12, True, "poor baby"),
    (13, True, "did you eat"),
    (14, False, "not yet"),
    (15, True, "go eat something"),
    (16, True, "im serious"),
]

SESSION_2 = [
    (0, True, "you forgot to call me back last night"),
    (2, False, "sorry i fell asleep on the couch"),
    (3, True, "mhm"),
    (4, True, "sure you did"),
    (6, False, "i actually did"),
    (7, True, "whatever"),
    (9, True, "im not even mad"),
    (10, True, "ok maybe a little"),
    (11, False, "ill make it up to you"),
    (12, True, "you better"),
]

SESSION_3 = [
    (0, False, "did you see that thing i sent you"),
    (1, True, "the video?"),
    (2, True, "yes i watched it like 4 times"),
    (3, False, "it was so dumb"),
    (4, True, "i know but the ending got me"),
    (6, True, "we should do that this weekend"),
    (7, False, "do what"),
    (8, True, "the thing with the boats obviously"),
    (9, False, "oh"),
    (10, False, "yeah ok"),
    (11, True, "dont sound so excited lol"),
]


def build_rows():
    rows = []
    counter = 0
    base = datetime(2025, 3, 14, 19, 0, 0)

    for day_offset, session in enumerate([SESSION_1, SESSION_2, SESSION_3]):
        session_start = base + timedelta(days=day_offset * 11)
        for minutes, is_her, text in session:
            counter += 1
            ts = session_start + timedelta(minutes=minutes)
            rows.append(
                {
                    "id": counter,
                    "MsgSvrID": f"msg-{counter:05d}",
                    "type_name": "text",
                    "is_sender": 1 if is_her else 0,
                    "talker": HER if is_her else HIM,
                    "room_name": ROOM,
                    "msg": text,
                    "src": "",
                    "CreateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_forward": "",
                }
            )
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    her = sum(1 for r in rows if r["is_sender"] == 1)
    print(f"Wrote {len(rows)} rows ({her} hers / {len(rows) - her} his) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
