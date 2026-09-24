"""Intent layer: transcript -> emotion keyframes for Audio2Face (the "prompted" route).

    python analysis/intent.py samples/clip.wav                 # uses samples/clip.words.json
    python analysis/intent.py samples/clip.wav --redo          # ignore the cached intent.json

1. An LLM reads the word-timed transcript (transcribe.py) and assigns each sentence a mix of
   A2F's explicit emotions (amazement, anger, cheekiness, disgust, fear, grief, joy,
   outofbreath, pain, sadness), 0..1, mostly subtle. Backend: project-local Ollama
   (qwen2.5:7b, models on D:, port 11435). The reply is cached as <clip>.intent.json — hand-editable, and
   re-running uses the cache.
2. The per-sentence targets become a smooth 30 fps curve (eased transitions, relax to neutral
   in long pauses) written as <clip>.emotion.txt, which rigstudio-a2f reads next to the wav.

The emotion values here are authored from the *text*; Audio2Emotion is not involved.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EMOTIONS = ["amazement", "anger", "cheekiness", "disgust", "fear", "grief", "joy", "outofbreath", "pain", "sadness"]
FPS = 30
EASE_IN, EASE_OUT, RELAX_GAP = 0.45, 0.8, 2.5   # seconds

# Graydient's prompt parser gives {a|b}, (word:1.2), [..] and / special meaning, so the prompt
# and the requested reply use none of them: plain lines in, plain lines out.
PROMPT = """You are the facial-performance director for an animated talking-head host.
Below is a timed transcript, one sentence per line, formatted as: S<number> at <start seconds>s - <text>
For EACH sentence decide the emotional colouring of the speaker's face while saying it.

Allowed emotions: amazement, anger, cheekiness, disgust, fear, grief, joy, outofbreath, pain, sadness.
Guidance:
- Talking heads are mostly subtle: typical weights 0.1 to 0.4. Use more than 0.6 only for genuinely strong moments.
- At most 2 emotions per sentence. A neutral sentence is written as none.
- cheekiness means playful, wry, sarcastic or teasing. amazement means surprise, wonder or excitement.
  joy means warmth, pleasure or enthusiasm. disgust means distaste, contempt or skepticism.
- Follow the arc: emotions should carry over and shift gradually, not flip every sentence.

Reply with exactly one line per sentence, S0 first, and nothing else. Each line is the sentence
label followed by the word none, or by one or two emotion names each followed by its weight.
Pick weights freely between 0.05 and 1. Illustrations from a different transcript:
S97 disgust 0.15
S98 none
S99 amazement 0.6 fear 0.1

Transcript:
LINES"""

_UNSAFE = re.compile(r"[{}\[\]()|/\<>:]")


OLLAMA_URL = os.environ.get("RIGSTUDIO_OLLAMA", "http://127.0.0.1:11435")   # project-local server


def ask_llm(sentences, log, backend="ollama", model="qwen2.5:7b"):
    lines = "\n".join(f"S{i} at {s['t0']:.1f}s - {_UNSAFE.sub(' ', s['text'])}" for i, s in enumerate(sentences))
    prompt = PROMPT.replace("LINES", lines)
    if backend == "ollama":
        import requests
        r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=900, json={
            "model": model, "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 8192},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["message"]["content"]
    # graydient lyric-llm: currently ignores API prompts (returns "{}"), kept for when it's fixed
    from graydient_api import decode_data_image, render
    _, urls = render(prompt, "/run:lyric-llm", log=log)
    last = None
    for u in reversed(urls):          # the data image is normally the last output
        try:
            return decode_data_image(u)
        except Exception as e:        # the card image is not a data image
            last = e
    raise RuntimeError(f"no decodable LLM output: {last}")


def parse_reply(text, n):
    """Lines like 'S3 joy 0.3 cheekiness 0.2' or 'S4 none' -> per-sentence dicts."""
    out, seen = [{} for _ in range(n)], 0
    for line in text.splitlines():
        m = re.match(r"\s*S(\d+)\b(.*)", line)
        if not m or int(m.group(1)) >= n:
            continue
        seen += 1
        pairs = re.findall(r"([a-z]+)\s+([01](?:\.\d+)?)", m.group(2).lower())
        out[int(m.group(1))] = {k: float(min(1.0, float(v))) for k, v in pairs if k in EMOTIONS}
    if not seen:
        raise ValueError(f"no 'S<n> ...' lines in LLM reply: {text[:300]!r}")
    return out


def emotion_curve(sentences, per_sentence, duration):
    """Per-sentence targets -> dense (frames, 10). Ease into each target just before the sentence,
    hold, and relax toward neutral when a pause is longer than RELAX_GAP."""
    frames = int(np.ceil(duration * FPS)) + 1
    t = np.arange(frames) / FPS
    keys = [(0.0, np.zeros(len(EMOTIONS)))]
    for i, (s, e) in enumerate(zip(sentences, per_sentence)):
        v = np.array([e.get(k, 0.0) for k in EMOTIONS])
        keys.append((max(0.0, s["t0"] - EASE_IN), keys[-1][1]))          # start of transition
        keys.append((s["t0"] + 0.05, v))                                  # target reached
        nxt = sentences[i + 1]["t0"] if i + 1 < len(sentences) else duration + 10
        if nxt - s["t1"] > RELAX_GAP:
            keys.append((s["t1"] + 0.3, v))
            keys.append((s["t1"] + 0.3 + EASE_OUT, np.zeros(len(EMOTIONS))))
        else:
            keys.append((s["t1"], v))
    keys.sort(key=lambda kv: kv[0])
    kt = np.array([k[0] for k in keys])
    kv = np.stack([k[1] for k in keys])
    out = np.zeros((frames, len(EMOTIONS)))
    for j in range(len(EMOTIONS)):
        lin = np.interp(t, kt, kv[:, j])
        out[:, j] = lin
    # smoothstep-ish easing: light temporal smoothing of the piecewise-linear ramps
    from scipy.ndimage import gaussian_filter1d
    return t, np.clip(gaussian_filter1d(out, sigma=0.12 * FPS, axis=0), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--words")
    ap.add_argument("--redo", action="store_true", help="ask the LLM again")
    ap.add_argument("--reparse", action="store_true", help="re-parse the saved raw reply")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "graydient"])
    ap.add_argument("--model", default="qwen2.5:7b")
    a = ap.parse_args()
    base = os.path.splitext(a.audio)[0]
    words = json.load(open(a.words or base + ".words.json", encoding="utf-8"))
    sents = words["sentences"]
    ipath = base + ".intent.json"
    if os.path.isfile(ipath) and not a.redo and not a.reparse:
        per = json.load(open(ipath, encoding="utf-8"))["per_sentence"]
        print(f"using cached {ipath}")
    else:
        raw = base + ".intent.raw.txt"
        if os.path.isfile(raw) and not a.redo:
            reply = open(raw, encoding="utf-8").read()
        else:
            reply = ask_llm(sents, print, a.backend, a.model)
            with open(raw, "w", encoding="utf-8") as f:     # keep the reply even if parsing fails
                f.write(reply)
        per = parse_reply(reply, len(sents))
        print(f"LLM reply ({len(reply)} chars), first lines:\n  " + "\n  ".join(reply.splitlines()[:5]))
        with open(ipath, "w", encoding="utf-8") as f:
            json.dump({"backend": f"{a.backend} {a.model}", "emotions": EMOTIONS,
                       "raw_reply": reply,
                       "per_sentence": per,
                       "sentences": [{"i": i, "t0": s["t0"], "t1": s["t1"], "text": s["text"], "e": e}
                                     for i, (s, e) in enumerate(zip(sents, per))]},
                      f, ensure_ascii=False, indent=1)
    import soundfile as sf
    duration = sf.info(a.audio).duration
    t, E = emotion_curve(sents, per, duration)
    with open(base + ".emotion.txt", "w", newline="\n") as f:
        for ti, row in zip(t, E):
            f.write(f"{ti:.4f} " + " ".join(f"{x:.4f}" for x in row) + "\n")
    used = {k: round(float(E[:, i].max()), 2) for i, k in enumerate(EMOTIONS) if E[:, i].max() > 0.01}
    print(f"{base}.emotion.txt: {len(t)} frames; peak per emotion {used}")
    for s, e in list(zip(sents, per))[:12]:
        print(f"  {s['t0']:6.1f}s {json.dumps(e):40} {s['text'][:70]}")


if __name__ == "__main__":
    main()
