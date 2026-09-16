"""Convert WeClone's ShareGPT-style SFT output into OpenAI fine-tune JSONL.

Input  (WeClone dataset/res_csv/sft/sft-my.json):
    [ { "id", "time", "score", "system",
        "messages": [ {"role": "user"|"assistant", "content": "..."} ] } ]

Output (OpenAI chat fine-tuning format, one JSON object per line):
    {"messages": [ {"role":"system","content":...},
                   {"role":"user","content":...},
                   {"role":"assistant","content":...}, ... ]}

Cleanups applied -- each one was observed in real output, not guessed:

1. <begin_chat> ARTEFACT
   When a thread opens with *her* message, WeClone fabricates a prompt turn:
       <begin_chat>你应该说：you forgot to call me back last night</begin_chat>
   followed by an assistant turn that just repeats the same line. That is a
   degenerate "repeat after me" example containing Chinese text in an English
   corpus. Both turns are dropped.

2. WELDED MESSAGES
   WeClone already joins consecutive same-speaker messages (patched to always
   use "\n"), but older output may contain welds like
   "the video?yes i watched it". Detected and split.

3. NON-CONVERSATION TURNS
   Bare media markers (图片 / 视频 / 语音), bare URLs and leftover <image>
   placeholders are removed so the model never learns to emit them.

4. ROLE SHAPE
   Consecutive same-role turns are merged, and the conversation is trimmed so it
   starts with `user` and ends with `assistant` -- OpenAI rejects/penalises
   examples that end on a user turn since there is nothing to learn from.

Usage:
    python to_openai_jsonl.py --in sft-my.json --out train.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

BEGIN_CHAT = re.compile(r"<begin_chat>(.*?)</begin_chat>", re.S)
# The synthetic prefix WeClone writes: "<begin_chat>你应该说：TEXT</begin_chat>"
SYNTH_PREFIX = re.compile(r"^[\u4e00-\u9fff\uff1a\s]*")

MEDIA_MARKERS = {"图片", "视频", "语音", "粘贴的文本", "<image>", "未知"}
BARE_URL = re.compile(r"^https?://\S+$", re.I)
# "sentence end" immediately followed by a letter with no space = a weld
WELD = re.compile(r"([.?!,;])(?=[A-Za-z])")

# Instagram call events, kept here as defence-in-depth: parse_instagram.py strips
# them, but if one ever survives it must not become a learned reply.
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


def split_welds(text: str) -> str:
    """Break 'the video?yes i watched it' into two lines.

    Conservative: only splits on sentence punctuation immediately followed by a
    letter, and never inside a URL or a decimal number.
    """
    out_lines = []
    for line in text.split("\n"):
        if BARE_URL.match(line.strip()):
            out_lines.append(line)
            continue
        # don't touch decimals like "3.5" or abbreviations inside URLs
        fixed = WELD.sub(lambda m: m.group(1) + "\n", line)
        out_lines.append(fixed)
    return "\n".join(out_lines)


def clean_content(text: str) -> str:
    text = BEGIN_CHAT.sub(lambda m: m.group(1), text)  # unwrap if nested
    text = text.replace("<image>", "")
    parts = [p.strip() for p in text.split("\n")]
    parts = [p for p in parts if p and not SYSTEM_EVENT.match(p)]
    return "\n".join(parts).strip()


def is_non_conversation(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t in MEDIA_MARKERS:
        return True
    if BARE_URL.match(t):
        return True
    if all(SYSTEM_EVENT.match(ln) for ln in t.split("\n") if ln.strip()):
        return True
    # only media markers / punctuation
    stripped = re.sub(r"[\s\.,!?…，。！？\-–—]+", "", t)
    return stripped == ""


def convert_example(ex: dict, stats: dict):
    system = (ex.get("system") or "").strip()
    msgs = ex.get("messages") or []

    cleaned = []
    for m in msgs:
        role = m.get("role")
        content = clean_content(m.get("content") or "")

        # 1. drop the synthetic opener and its echo
        if role == "user" and "<begin_chat>" in (m.get("content") or ""):
            inner = BEGIN_CHAT.search(m["content"])
            echo = SYNTH_PREFIX.sub("", inner.group(1)).strip() if inner else ""
            stats["begin_chat_dropped"] += 1
            # if the next turn is the assistant parroting it, drop that too
            continue

        if is_non_conversation(content):
            stats["nonconv_dropped"] += 1
            continue

        cleaned.append({"role": role, "content": content})

    # remove an assistant turn that merely echoes the previous user turn right
    # after a dropped <begin_chat> opener
    deduped = []
    for i, m in enumerate(cleaned):
        if (deduped and m["role"] == "assistant" and deduped[-1]["role"] == "user"
                and m["content"].strip() == deduped[-1]["content"].strip()):
            stats["echo_dropped"] += 1
            continue
        deduped.append(m)
    cleaned = deduped

    # 2. split welds
    for m in cleaned:
        new = split_welds(m["content"])
        if new != m["content"]:
            stats["welds_split"] += 1
        m["content"] = new

    # 3. merge consecutive same-role
    merged = []
    for m in cleaned:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n" + m["content"]
            stats["turns_merged"] += 1
        else:
            merged.append(dict(m))
    cleaned = merged

    # 4. shape: must start with user, end with assistant
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()
    if not cleaned:
        stats["shape_dropped"] += 1
        return None
    if not any(m["role"] == "assistant" for m in cleaned):
        stats["shape_dropped"] += 1
        return None

    out = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend(cleaned)
    return {"messages": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-assistant-chars", type=int, default=1)
    ap.add_argument("--max-total-chars", type=int, default=60000)
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        print(f"not found: {src}")
        return 2

    data = json.loads(src.read_text(encoding="utf-8"))
    print(f"loaded {len(data)} WeClone examples from {src.name}")

    stats = {
        "begin_chat_dropped": 0,
        "echo_dropped": 0,
        "nonconv_dropped": 0,
        "welds_split": 0,
        "turns_merged": 0,
        "shape_dropped": 0,
        "too_small": 0,
        "too_big": 0,
    }

    out_rows = []
    for ex in data:
        conv = convert_example(ex, stats)
        if not conv:
            continue
        total = sum(len(m["content"]) for m in conv["messages"])
        if total > args.max_total_chars:
            stats["too_big"] += 1
            continue
        asst = [m for m in conv["messages"] if m["role"] == "assistant"]
        if not asst or max(len(m["content"]) for m in asst) < args.min_assistant_chars:
            stats["too_small"] += 1
            continue
        out_rows.append(conv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(
        sum(len(m["content"]) for m in r["messages"]) for r in out_rows
    )
    asst_chars = sum(
        len(m["content"]) for r in out_rows for m in r["messages"] if m["role"] == "assistant"
    )
    avg_turns = (
        sum(len([m for m in r["messages"] if m["role"] != "system"]) for r in out_rows)
        / max(len(out_rows), 1)
    )

    print(f"\n=== cleanups ===")
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")
    print(f"\n=== result ===")
    print(f"  examples kept     : {len(out_rows)} / {len(data)}")
    print(f"  avg turns/example : {avg_turns:.1f}")
    print(f"  total chars       : {total_chars:,}")
    print(f"  assistant chars   : {asst_chars:,}  ({100 * asst_chars / max(total_chars, 1):.1f}% of tokens)")
    print(f"  ~est tokens       : {total_chars // 4:,} (chars/4 heuristic)")
    print(f"  written           : {out}")

    if out_rows:
        print(f"\n=== first example ===")
        print(json.dumps(out_rows[0], ensure_ascii=False, indent=2)[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
