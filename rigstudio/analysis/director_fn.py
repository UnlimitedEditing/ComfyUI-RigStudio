"""Director v3: the LLM labels each sentence's FUNCTION (reading comprehension); a table we own maps
function -> delivery (acting taste). Small models are unreliable at taste but decent at comprehension.

    python analysis/director_fn.py samples/real1.wav --lyrics samples/real1.medium.lyrics.json \
        [--model qwen2.5:7b] [--brief "..."] [-o out.json]

The table below was written from general acting sense BEFORE scoring, not fitted to the benchmark.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from director import ask  # noqa: E402

from vocab import FUNCTIONS, TABLE  # noqa: E402,F401

PROMPT = """You label the FUNCTION of each sentence in a talking-head video script. Read for meaning.

Functions:
FUNCS

Already labelled (context):
DONE

NEW sentences:
LINES

For each NEW sentence write one line: S<number> - <why, at most 8 words> - <function>
Use exactly one function word from the list. Illustration from another script:
S41 - pokes fun at his own setup - aside
S42 - names the ingredients - list"""


def parse(reply, ids):
    got = {}
    for line in reply.splitlines():
        m = re.match(r"\s*s(\d+)\b.*-\s*([a-z]+)\s*$", line.strip().lower())
        if m and int(m.group(1)) in ids and m.group(2) in FUNCTIONS:
            got[int(m.group(1))] = m.group(2)
    return got


BRIEF_BLOCK = "About this script (from its author): BRIEF\n\n"


def direct(lyrics, model="qwen2.5:7b", window=8, table=None, brief="", known=None, seed=None):
    """brief: the author's plain-English note, shown to the LLM as context (empty = the benchmarked
    prompt, byte for byte; the brief itself is not benchmarked yet). known: {sentence index: function}
    for lines the user locked: they replace the LLM's label, so later windows see them as context."""
    table = table or TABLE
    known = known or {}
    sents = [{"t0": s["start"], "t1": s["end"], "text": s["text"]} for s in lyrics["timeline"]]
    funcs = "\n".join(f"- {k}: {v}" for k, v in FUNCTIONS.items())
    head = BRIEF_BLOCK.replace("BRIEF", " ".join(brief.split())) if brief.strip() else ""
    fn, replies = {}, []
    for w0 in range(0, len(sents), window):
        ids = list(range(w0, min(len(sents), w0 + window)))
        done = "\n".join(f"S{i} - {sents[i]['text'][:70]} -> {fn[i]}" for i in range(max(0, w0 - 6), w0)) \
            or "(start of script)"
        lines = "\n".join(f"S{i} - {sents[i]['text']}" for i in ids)
        reply = ask(head + PROMPT.replace("FUNCS", funcs).replace("DONE", done).replace("LINES", lines), model,
                    seed=None if seed is None else seed + w0)
        replies.append(reply)
        got = parse(reply, set(ids))
        for i in ids:
            fn[i] = known.get(i) or got.get(i, "fact")
    parsed = sum(len(parse(r, set(range(len(sents))))) for r in replies)
    return {"reply": "\n---\n".join(replies), "parsed_lines": parsed,
            "sentences": [{**s, "function": fn[i], "delivery": table[fn[i]][0], "intensity": table[fn[i]][1]}
                          for i, s in enumerate(sents)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    res = direct(json.load(open(a.lyrics, encoding="utf-8")), a.model)
    out = a.out or os.path.splitext(a.audio)[0] + ".direction.fn.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out}: parsed {res['parsed_lines']}/{len(res['sentences'])} lines")


if __name__ == "__main__":
    main()


def direct_voted(lyrics, model="qwen2.5:7b", runs=3, table=None):
    """Majority vote over several runs (ties -> first run), then deterministic rules:
    a sentence ending in '?' is a question."""
    from collections import Counter
    table = table or TABLE
    results = [direct(lyrics, model, table=table) for _ in range(runs)]
    out = results[0]
    for i, s in enumerate(out["sentences"]):
        votes = Counter(r["sentences"][i]["function"] for r in results)
        top = max(votes.values())
        fn = next(r["sentences"][i]["function"] for r in results if votes[r["sentences"][i]["function"]] == top)
        if s["text"].rstrip().endswith("?"):
            fn = "question"
        s["function"], s["votes"] = fn, dict(votes)
        s["delivery"], s["intensity"] = table[fn]
    out["reply"] = "\n=====\n".join(r["reply"] for r in results)
    return out
