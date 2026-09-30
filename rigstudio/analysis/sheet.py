"""Direction sheet: everything Perform needs to reproduce a performance, in one M3DS PNG.

    python analysis/sheet.py make samples/real1.wav --lyrics samples/real1.medium.lyrics.json \
        [--direction bench/runs-real1-fn14b.json] [--audio-url URL] [--brief "..."] [--seed N] [-o out.png]
    python analysis/sheet.py from-bench bench/real1.direction.json --lyrics samples/real1.medium.lyrics.json \
        [--audio samples/real1.wav] [-o bench/real1.sheet.png]
    python analysis/sheet.py show sheet.png [--json]

The PNG carries `direction.rigstudio.json` (M3DS, byte-compatible with Meshsmuggler / YuE2 score
PNGs). Sheet Studio (studio/index.html) decodes, edits and re-encodes the same file; the sheet embeds
the vocabulary + function table it was directed with, so Studio needs no hard-coded lists.

Sentence fields: i, t0, t1, text, words [[t0, t1, word], ...], function (null until directed),
delivery, intensity (null when neutral), locked, source (none | llm | user).
Locked lines are the user's: re-directing (apply_direction) never touches them.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ds  # noqa: E402
from vocab import INTENSITY, TABLE, VOCAB, perform_weights, vocabulary  # noqa: E402

FORMAT = "rigstudio.direction"
VERSION = 1
FILENAME = "direction.rigstudio.json"
A2F_MODEL = "mark-v2.3"
A2F_ENGINE = "a2f-engine-v1"
MIN_SENTENCE = 0.4  # seconds


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_sheet(lyrics, audio_url="", audio_sha256="", brief="", seed=0, a2f_model=A2F_MODEL,
              a2f_engine=A2F_ENGINE):
    """Undirected sheet from a sentence-level transcript (RigStudioTranscribe's lyrics_json)."""
    sents = []
    for t in lyrics["timeline"]:
        if t.get("type", "lyric") != "lyric" or not t["text"].strip():
            continue
        sents.append({"i": len(sents), "t0": round(t["start"], 3), "t1": round(t["end"], 3), "text": t["text"].strip(),
                      "words": [[round(w["start"], 3), round(w["end"], 3), w["text"].strip()] for w in t.get("words", [])],
                      "function": None, "delivery": "neutral", "intensity": None, "locked": False, "source": "none"})
    # Whisper can stamp a short interjection with zero length ("See?" at 112.98..112.98 in real1).
    # Widen it into the following gap so it can be played and performed.
    for k, s in enumerate(sents):
        if s["t1"] - s["t0"] < MIN_SENTENCE:
            limit = sents[k + 1]["t0"] if k + 1 < len(sents) else s["t0"] + MIN_SENTENCE
            s["t1"] = round(max(s["t1"], min(s["t0"] + MIN_SENTENCE, limit)), 3)
            if s["words"] and s["words"][-1][1] < s["t1"]:
                s["words"][-1][1] = s["t1"]
    return {"format": FORMAT, "version": VERSION, "created": _now(),
            "audio": {"url": audio_url, "sha256": audio_sha256, "duration": lyrics.get("duration")},
            "brief": brief, "seed": int(seed),
            "transcript": {"language": lyrics.get("language"), "transcriber": lyrics.get("transcriber")},
            "sentences": sents, "direction": None, "vocabulary": vocabulary(),
            "a2f": {"model": a2f_model, "engine": a2f_engine}}


def _same_text(a, b):
    norm = lambda s: " ".join(s.lower().split())
    return norm(a) == norm(b)


def apply_direction(sheet, directed, director=None):
    """Fill function/delivery/intensity from a director result (list aligned to the sheet's
    sentences) on every UNLOCKED line. Returns the number of locked lines kept."""
    sents = sheet["sentences"]
    if len(directed) != len(sents):
        raise ValueError(f"direction has {len(directed)} sentences, sheet has {len(sents)}: different transcript")
    for s, d in zip(sents, directed):
        if "text" in d and not _same_text(s["text"], d["text"]):
            raise ValueError(f"sentence {s['i']} text differs: {s['text'][:40]!r} vs {d['text'][:40]!r}")
    kept = 0
    for s, d in zip(sents, directed):
        if s["locked"]:
            kept += 1
            continue
        s["function"], s["delivery"], s["intensity"] = d.get("function"), d["delivery"], d.get("intensity")
        s["source"] = "llm"
    sheet["direction"] = {**(director or {}), "at": _now(), "kept_locked": kept}
    sheet["vocabulary"] = vocabulary()
    return kept


def set_line(sheet, i, delivery=None, intensity=None, function=None):
    """A user edit (what Sheet Studio does): picking a function applies the table; any edit locks."""
    s = sheet["sentences"][i]
    if function is not None:
        s["function"] = function
        s["delivery"], s["intensity"] = TABLE[function]
    if delivery is not None:
        s["delivery"] = delivery
        if delivery == "neutral":
            s["intensity"] = None
        elif not s["intensity"]:
            s["intensity"] = "clear"
    if intensity is not None and s["delivery"] != "neutral":
        s["intensity"] = intensity
    s["locked"], s["source"] = True, "user"


def validate(sheet):
    issues = []
    if sheet.get("format") != FORMAT:
        return [f"not a direction sheet (format {sheet.get('format')!r})"]
    if sheet.get("version") != VERSION:
        issues.append(f"sheet version {sheet.get('version')} (this code reads {VERSION})")
    voc = sheet.get("vocabulary") or {}
    if voc.get("sha") != vocabulary()["sha"]:
        issues.append(f"vocabulary {voc.get('version')}/{voc.get('sha')} differs from current "
                      f"{vocabulary()['version']}/{vocabulary()['sha']}")
    prev = -1.0
    for s in sheet.get("sentences", []):
        tag = f"S{s.get('i')}"
        if s.get("delivery") not in VOCAB:
            issues.append(f"{tag}: unknown delivery {s.get('delivery')!r}")
        if s.get("delivery") == "neutral" and s.get("intensity"):
            issues.append(f"{tag}: neutral with intensity {s['intensity']!r}")
        if s.get("delivery") not in (None, "neutral") and s.get("intensity") not in INTENSITY:
            issues.append(f"{tag}: bad intensity {s.get('intensity')!r}")
        if s.get("function") is not None and s["function"] not in TABLE:
            issues.append(f"{tag}: unknown function {s['function']!r}")
        if not (s.get("t1", 0) > s.get("t0", 0) >= prev - 0.05):
            issues.append(f"{tag}: times out of order ({s.get('t0')}..{s.get('t1')})")
        prev = s.get("t1", prev)
    return issues


def dumps(sheet):
    return json.dumps(sheet, ensure_ascii=False, indent=1)


def to_png(sheet):
    return m3ds.encode(dumps(sheet).encode("utf-8"), FILENAME)


def save_png(sheet, path):
    to_png(sheet).save(path, format="PNG", compress_level=6)


def loads(data):
    """PNG or JSON bytes -> sheet dict."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        data, _ = m3ds.decode_png_bytes(data)
    sheet = json.loads(data.decode("utf-8"))
    if sheet.get("format") != FORMAT:
        raise ValueError(f"not a Rig Studio direction sheet (format {sheet.get('format')!r})")
    return sheet


def load(path):
    with open(path, "rb") as f:
        return loads(f.read())


def from_benchmark(key, lyrics, **kw):
    """Approved benchmark key -> sheet with every line locked (source user)."""
    sheet = new_sheet(lyrics, **kw)
    apply_direction(sheet, key["sentences"], {"from": "benchmark", "status": key.get("status")})
    for s in sheet["sentences"]:
        s["locked"], s["source"] = True, "user"
    sheet["brief"] = kw.get("brief") or key.get("persona", "")
    return sheet


def lyrics_of(sheet):
    """The sheet's transcript in RigStudioTranscribe's lyrics_json shape (re-directing skips Whisper)."""
    return {"language": sheet["transcript"].get("language"), "duration": sheet["audio"].get("duration"),
            "transcriber": sheet["transcript"].get("transcriber"),
            "timeline": [{"start": s["t0"], "end": s["t1"], "text": s["text"], "type": "lyric",
                          "words": [{"start": a, "end": b, "text": w} for a, b, w in s["words"]]}
                         for s in sheet["sentences"]]}


def locked_functions(sheet):
    return {s["i"]: s["function"] for s in sheet["sentences"] if s["locked"] and s.get("function")}


_WH = re.compile(r"^(?:\W*(?:well|so|and|but|okay|ok|now|oh)\W+)*(what|why|how|where|who|whom|which|when)\b", re.I)


def to_intent(sheet):
    """Deterministic Perform input for RigStudioBuildTrack: S <t0> <t1> [q|wq] <emotion> <w> ... per line
    (letters, digits, dots and spaces only). Questions come from the text, as in RigStudioIntentPack."""
    parts = []
    for s in sheet["sentences"]:
        text = s["text"].rstrip()
        q = ("wq" if _WH.search(text) else "q") if text.endswith("?") else ""
        w = perform_weights(s["delivery"], s.get("intensity"))
        emo = " ".join(f"{e} {v}" for e, v in w.items()) or "none"
        parts.append(" ".join(x for x in ("S", f"{s['t0']:.2f}", f"{s['t1']:.2f}", q, emo) if x))
    return " ".join(parts)


def _summary(sheet):
    sents = sheet["sentences"]
    counts = {}
    for s in sents:
        counts[s["delivery"]] = counts.get(s["delivery"], 0) + 1
    d = sheet.get("direction") or {}
    return (f"{len(sents)} sentences, {sum(s['locked'] for s in sents)} locked, "
            f"audio {sheet['audio'].get('duration')} s sha {str(sheet['audio'].get('sha256'))[:12] or '-'}, "
            f"seed {sheet['seed']}, vocab v{sheet['vocabulary']['version']}/{sheet['vocabulary']['sha']}, "
            f"a2f {sheet['a2f']['model']}/{sheet['a2f']['engine']}, directed by {d.get('model') or d.get('from') or '-'}\n"
            "  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("audio")
    mk.add_argument("--lyrics", required=True)
    mk.add_argument("--direction", help="director_fn output json (sentences aligned to the transcript)")
    mk.add_argument("--audio-url", default="")
    mk.add_argument("--brief", default="")
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("-o", "--out")
    fb = sub.add_parser("from-bench")
    fb.add_argument("key")
    fb.add_argument("--lyrics", required=True)
    fb.add_argument("--audio")
    fb.add_argument("--audio-url", default="")
    fb.add_argument("-o", "--out")
    sh = sub.add_parser("show")
    sh.add_argument("sheet")
    sh.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.cmd == "show":
        sheet = load(a.sheet)
        print(dumps(sheet) if a.json else _summary(sheet))
        for issue in validate(sheet):
            print("  ! " + issue)
        return
    lyrics = json.load(open(a.lyrics, encoding="utf-8"))
    if a.cmd == "make":
        sheet = new_sheet(lyrics, a.audio_url, sha256_file(a.audio), a.brief, a.seed)
        if a.direction:
            run = json.load(open(a.direction, encoding="utf-8"))
            apply_direction(sheet, run["sentences"], {"from": os.path.basename(a.direction), **run.get("director", {})})
        out = a.out or os.path.splitext(a.audio)[0] + ".sheet.png"
    else:
        key = json.load(open(a.key, encoding="utf-8"))
        audio = a.audio or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.key))), key.get("audio", ""))
        sha = sha256_file(audio) if os.path.isfile(audio) else ""
        sheet = from_benchmark(key, lyrics, audio_url=a.audio_url, audio_sha256=sha)
        out = a.out or os.path.splitext(a.key)[0].replace(".direction", "") + ".sheet.png"
    save_png(sheet, out)
    back = load(out)
    assert back == json.loads(dumps(sheet)), "round-trip mismatch"
    print(f"{out}: {os.path.getsize(out)} bytes, round-trip OK\n  {_summary(back)}")
    for issue in validate(back):
        print("  ! " + issue)


if __name__ == "__main__":
    main()
