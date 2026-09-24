"""Analysis side of RigStudioBuildTrack, run as a subprocess with the private analysis deps.

    python track_job.py emotion <intent.txt> <duration_s> <out.emotion.txt>
    python track_job.py finish  <wav16k> <a2f.txt> <intent.txt> <bs_skin.npz> <out.track.json> [--seed N]

Intent text (Job A output, prompt-parser safe: letters, digits, dots, spaces only):
    S <t0> <t1> [q|wq] <emotion> <weight> [<emotion> <weight>] | none   S ...
e.g. "S 0.00 1.63 joy 0.3 S 1.88 3.37 none S 38.10 40.50 wq amazement 0.5"
q / wq mark a question / wh-question sentence (drives the question gestures).
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "analysis"))
import analyze as A  # noqa: E402
from intent import EMOTIONS, emotion_curve  # noqa: E402


def parse_intent(text):
    sents = []
    for chunk in re.split(r"\bS\s+", " " + text.strip()):
        tok = chunk.split()
        if len(tok) < 2:
            continue
        try:
            t0, t1 = float(tok[0]), float(tok[1])
        except ValueError:
            continue
        rest, q = tok[2:], ""
        if rest and rest[0] in ("q", "wq"):
            q, rest = rest[0], rest[1:]
        e = {}
        for k, v in zip(rest[0::2], rest[1::2]):
            if k in EMOTIONS:
                try:
                    e[k] = float(np.clip(float(v), 0, 1))
                except ValueError:
                    pass
        sents.append({"t0": t0, "t1": t1, "e": e, "q": q})
    return sorted(sents, key=lambda s: s["t0"])


def cmd_emotion(intent_p, duration, out_p):
    sents = parse_intent(open(intent_p, encoding="utf-8").read())
    t, E = emotion_curve(sents, [s["e"] for s in sents], float(duration))
    with open(out_p, "w", newline="\n") as f:
        for ti, row in zip(t, E):
            f.write(f"{ti:.4f} " + " ".join(f"{x:.4f}" for x in row) + "\n")
    print(json.dumps({"sentences": len(sents), "emotional": sum(1 for s in sents if s["e"]),
                      "questions": sum(1 for s in sents if s["q"])}))


def pose_names(npz, n):
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in np.load(npz, allow_pickle=True)["poseNames"]]
    if len(names) == n + 1 and names[0] == "neutral":
        names = names[1:]
    if len(names) != n:
        raise SystemExit(f"weight count {n} != {len(names)} pose names")
    return [re.sub(r"Right$", "_R", re.sub(r"Left$", "_L", x)) for x in names]


def cmd_finish(wav, a2f_p, intent_p, npz, out_p, seed, audio_name=""):
    au = A.analyse_audio(wav)
    frames = int(np.ceil(au["duration"] * A.FPS))
    phrases = A.phrases_from(au["speech"])
    sents = parse_intent(open(intent_p, encoding="utf-8").read())
    # question gestures come from the intent's q / wq markers (analyze.py reads sentence-final "?")
    words = {"sentences": [{"t0": s["t0"], "t1": s["t1"],
                            "text": {"q": "?", "wq": "what ?"}.get(s["q"], ".")} for s in sents]}
    rng = np.random.default_rng(seed)
    events = []
    idle = A.build_idle(au, phrases, frames, rng, "speaker", 0.0, events, idle_face=False)
    reflex = A.build_reflex_speaker(au, phrases, frames, rng, events, words)
    visemes = A.heuristic_visemes(au)

    with open(a2f_p) as f:
        n = int(f.readline())
        rows = np.array([list(map(float, l.split())) for l in f if l.strip()])
    grid = np.arange(frames) / A.FPS
    t = rows[:, 0] / 16000.0
    a2f = {nm: [round(float(x), 3) for x in np.interp(grid, t, np.clip(rows[:, i + 1], 0, 1))]
           for i, nm in enumerate(pose_names(npz, n))}
    for s in sents:
        if s["e"]:
            events.append({"t": round(s["t0"], 3), "kind": "emotion", "e": s["e"]})
    track = {
        "format": "rigstudio.expression_track", "version": 0, "fps": A.FPS, "frames": frames,
        "duration": round(au["duration"], 3), "role": "speaker",
        "source": {"audio": audio_name or os.path.basename(wav),
                   "generator": "ComfyUI-RigStudio RigStudioBuildTrack (hybrid)", "mouth": "a2f",
                   "a2f": "Audio2Face-3D v2.3 Mark + intent emotion keyframes", "visemes": "heuristic"},
        "layers": {"a2f": {"blend": "add", "weight": 1.0, "curves": a2f},
                   "idle": idle.export(), "reflex": reflex.export()},
        "visemes": visemes,
        "events": sorted(events, key=lambda e: e["t"]),
    }
    with open(out_p, "w") as f:
        json.dump(track, f, separators=(",", ":"))
    print(json.dumps({"frames": frames, "duration": track["duration"], "a2f_weights": n,
                      "events": len(events), "bytes": os.path.getsize(out_p)}))


if __name__ == "__main__":
    if sys.argv[1] == "emotion":
        cmd_emotion(*sys.argv[2:5])
    elif sys.argv[1] == "finish":
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
        name = sys.argv[sys.argv.index("--audio-name") + 1] if "--audio-name" in sys.argv else ""
        cmd_finish(*sys.argv[2:7], seed, name)
    else:
        raise SystemExit(__doc__)
