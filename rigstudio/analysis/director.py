"""Director v2: transcript (+ brief) (+ voice cues) -> per-sentence named delivery.

    python analysis/director.py samples/real1.wav --lyrics samples/real1.medium.lyrics.json \
        [--brief "..."] [--no-cues] [--no-persona] [--model qwen2.5:7b] [-o out.json]

Two passes against an LLM (project-local Ollama by default):
  1. persona + emotional arc for the whole script (one paragraph)
  2. one line per sentence: S<n> <delivery> <intensity>, from the fixed vocabulary below
Voice cues are OUR prosody measurements (loudness / pitch range / pace vs the speaker's own median,
pause before) rendered as words — not an emotion classifier, and not Audio2Emotion.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import INTENSITY, VOCAB  # noqa: E402,F401
OLLAMA = os.environ.get("RIGSTUDIO_OLLAMA", "http://127.0.0.1:11435")

PERSONA_PROMPT = """You are a performance director for an animated talking-head host.
Read the whole script below and write ONE short paragraph: who the speaker is, their overall attitude,
and how the emotional arc moves through the piece (which parts are pitch, joke, instruction, payoff).
BRIEF_BLOCK
Script:
SCRIPT"""

LINES_PROMPT = """You are a performance director for an animated talking-head host. Decide the delivery
of EACH sentence so the face can act it. Be a good actor: react to meaning, jokes, asides and payoffs,
but leave plain information neutral.

Speaker and arc: PERSONA
BRIEF_BLOCK
Allowed deliveries (use these words only):
VOCAB
Intensity: slight, clear or strong.
CUES_NOTE
Reply with exactly one line per sentence, S0 first, nothing else, as:
S<number> <delivery> <intensity>
Illustration from a different script: "S41 wry clear", "S42 neutral", "S43 delighted strong".

Sentences:
LINES"""


def ask(prompt, model, temperature=0.2, seed=None):
    body = {"model": model, "stream": False, "options": {"temperature": temperature, "num_ctx": 4096},
            "messages": [{"role": "user", "content": prompt}]}
    if seed is not None:
        body["options"]["seed"] = int(seed)   # same sampling, but reproducible
    if model.startswith("qwen3"):
        body["think"] = False          # Qwen3 reasons aloud by default; the line format needs plain output
    r = requests.post(f"{OLLAMA}/api/chat", timeout=5400, json=body)
    r.raise_for_status()
    return re.sub(r"<think>.*?</think>", "", r.json()["message"]["content"], flags=re.S).strip()


def voice_cues(wav, sents):
    """Per sentence, relative to the speaker's own medians: loudness, pitch range, pace, pause before."""
    import analyze as A
    au = A.analyse_audio(wav)
    af = A.AFPS
    rows = []
    for i, s in enumerate(sents):
        a, b = int(s["t0"] * af), max(int(s["t0"] * af) + 1, int(s["t1"] * af))
        db = au["db"][a:b][au["speech"][a:b]] if au["speech"][a:b].any() else au["db"][a:b]
        st = au["st"][a:b]
        st = st[np.isfinite(st)]
        words = max(1, len(s["text"].split()))
        rows.append({"loud": float(np.median(db)) if len(db) else np.nan,
                     "range": float(np.percentile(st, 90) - np.percentile(st, 10)) if len(st) > 5 else np.nan,
                     "rate": words / max(0.3, s["t1"] - s["t0"]),
                     "pause": s["t0"] - (sents[i - 1]["t1"] if i else 0.0)})
    med = {k: np.nanmedian([r[k] for r in rows]) for k in ("loud", "range", "rate")}
    out = []
    for r in rows:
        c = []
        if r["loud"] - med["loud"] > 3: c.append("louder")
        elif r["loud"] - med["loud"] < -3: c.append("quieter")
        if r["range"] > med["range"] * 1.4: c.append("lively pitch")
        elif r["range"] < med["range"] * 0.6: c.append("flat pitch")
        if r["rate"] > med["rate"] * 1.25: c.append("fast")
        elif r["rate"] < med["rate"] * 0.75: c.append("slow")
        if r["pause"] > 1.0: c.append("pause before")
        out.append(", ".join(c))
    return out


def parse(reply, n):
    out = [None] * n
    for line in reply.splitlines():
        m = re.match(r"\s*s(\d+)\b\s+([a-z]+)(?:\s+([a-z]+))?", line.strip().lower())
        if m and int(m.group(1)) < n and m.group(2) in VOCAB:
            inten = m.group(3) if m.group(3) in INTENSITY else None
            out[int(m.group(1))] = (m.group(2), None if m.group(2) == "neutral" else (inten or "clear"))
    return [x or ("neutral", None) for x in out], sum(x is not None for x in out)


WINDOW_PROMPT = """You are a performance director for an animated talking-head host. Decide how the face
should act each NEW sentence. Good acting reacts to meaning: jokes, asides, questions, sales pitch,
payoffs, reassurance. Plain information (lists, steps, facts) stays neutral. Do not repeat one delivery
out of habit: judge every sentence on its own words.

Speaker and arc: PERSONA
BRIEF_BLOCK
Allowed deliveries:
VOCAB
Intensity: slight, clear or strong.
CUES_NOTE
Already decided (context, do not repeat):
DONE

NEW sentences:
LINES

For each NEW sentence write one line: S<number> - <why, at most 8 words> - <delivery> <intensity>
(write neutral without intensity). Illustration from another script:
S41 - mocking his own mistake - wry clear
S42 - reading an ingredient list - neutral"""


def parse_window(reply, ids):
    got = {}
    for line in reply.splitlines():
        m = re.match(r"\s*s(\d+)\b.*-\s*([a-z]+)(?:\s+(slight|clear|strong))?\s*$", line.strip().lower())
        if m and int(m.group(1)) in ids and m.group(2) in VOCAB:
            d = m.group(2)
            got[int(m.group(1))] = (d, None if d == "neutral" else (m.group(3) or "clear"))
    return got


def direct_windowed(wav, lyrics, model="qwen2.5:7b", brief="", cues=True, persona=True, window=8):
    """Reason-then-label in small windows, earlier decisions shown as context (breaks mode collapse)."""
    sents = [{"t0": s["start"], "t1": s["end"], "text": s["text"]} for s in lyrics["timeline"]]
    brief_block = f"Director's brief from the creator: {brief}\n" if brief else ""
    script = "\n".join(s["text"] for s in sents)
    arc = (ask(PERSONA_PROMPT.replace("BRIEF_BLOCK", brief_block).replace("SCRIPT", script), model)
           if persona else "(not provided)")
    cue_words = voice_cues(wav, sents) if cues else [""] * len(sents)
    vocab = "\n".join(f"- {k}: {v}" for k, v in VOCAB.items())
    cues_note = ("Voice notes in [brackets] describe how the line was actually spoken, relative to this "
                 "speaker's normal delivery. Use them as evidence.\n") if cues else ""
    decided, replies, parsed = {}, [], 0
    for w0 in range(0, len(sents), window):
        ids = list(range(w0, min(len(sents), w0 + window)))
        done = "\n".join(f"S{i} - {sents[i]['text'][:70]} -> {decided[i][0]} {decided[i][1] or ''}".rstrip()
                         for i in range(max(0, w0 - 6), w0)) or "(start of script)"
        lines = "\n".join(f"S{i} - {sents[i]['text']}" + (f"   [voice: {cue_words[i]}]" if cue_words[i] else "")
                          for i in ids)
        prompt = (WINDOW_PROMPT.replace("PERSONA", arc).replace("BRIEF_BLOCK", brief_block).replace("VOCAB", vocab)
                  .replace("CUES_NOTE", cues_note).replace("DONE", done).replace("LINES", lines))
        reply = ask(prompt, model)
        replies.append(reply)
        got = parse_window(reply, set(ids))
        parsed += len(got)
        for i in ids:
            decided[i] = got.get(i, ("neutral", None))
    return {"persona": arc, "cues": cue_words, "reply": "\n---\n".join(replies), "parsed_lines": parsed,
            "sentences": [{**s, "delivery": decided[i][0], "intensity": decided[i][1]} for i, s in enumerate(sents)]}


def direct(wav, lyrics, model="qwen2.5:7b", brief="", cues=True, persona=True):
    sents = [{"t0": s["start"], "t1": s["end"], "text": s["text"]} for s in lyrics["timeline"]]
    brief_block = f"Director's brief from the creator: {brief}\n" if brief else ""
    script = "\n".join(s["text"] for s in sents)
    arc = ask(PERSONA_PROMPT.replace("BRIEF_BLOCK", brief_block).replace("SCRIPT", script), model) if persona \
        else "(not provided)"
    cue_words = voice_cues(wav, sents) if cues else [""] * len(sents)
    lines = "\n".join(f"S{i} - {s['text']}" + (f"   [voice: {c}]" if c else "")
                      for i, (s, c) in enumerate(zip(sents, cue_words)))
    prompt = (LINES_PROMPT.replace("PERSONA", arc).replace("BRIEF_BLOCK", brief_block)
              .replace("VOCAB", "\n".join(f"- {k}: {v}" for k, v in VOCAB.items()))
              .replace("CUES_NOTE", "Voice notes in [brackets] describe how the line was actually spoken, relative "
                                    "to this speaker's normal delivery. Use them as evidence.\n" if cues else "")
              .replace("LINES", lines))
    reply = ask(prompt, model)
    per, parsed = parse(reply, len(sents))
    return {"persona": arc, "cues": cue_words, "reply": reply, "parsed_lines": parsed,
            "sentences": [{**s, "delivery": d, "intensity": it} for s, (d, it) in zip(sents, per)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--brief", default="")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--no-cues", action="store_true")
    ap.add_argument("--no-persona", action="store_true")
    ap.add_argument("--windowed", action="store_true", help="reason-then-label in windows of 8")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    fn = direct_windowed if a.windowed else direct
    res = fn(a.audio, json.load(open(a.lyrics, encoding="utf-8")), a.model, a.brief,
             not a.no_cues, not a.no_persona)
    out = a.out or os.path.splitext(a.audio)[0] + ".direction.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out}: parsed {res['parsed_lines']}/{len(res['sentences'])} lines")


if __name__ == "__main__":
    main()
