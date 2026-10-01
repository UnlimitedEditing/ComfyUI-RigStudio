"""Blocking scripts -> rigstudio.shot. The deterministic half of director mode.

The LLM never writes coordinates, keys or eases. It writes a short BLOCKING SCRIPT, one beat per line, from a fixed
vocabulary; this module compiles it against the set (prop positions) and the dialogue (word timings) into shot keys,
moves and camera/lamp keys, and rejects anything it can't compile (reported, line skipped).

Script grammar (one line = one beat; blank lines and `#` comments ignored):

    <when> <subject> <verb> [args]

  when     0 | 4.2 | S2 | S2.end | S2:"word" | S2:word      (+0.3 / -0.2 offsets allowed: S2.end+0.5)
           Sn = the n-th sentence of the shot's dialogue (numbered across all lines, 1-based)
  subject  an actor id | a prop name | camera | lamp
  verbs    actor:  at <place>            place immediately (use at 0 for the opening position)
                   go <place> [slow|quick]   walk/float there, arriving after 0.6/1.2/2.2 s
                   face <camera|left|right|<actor>|<prop>>
                   hop|nod|shake|tilt|spin|squash|wobble [small|big]
           prop:   push <away|back|forward|left|right> [small|big]    (movable props only)
                   bump [small|big]
           camera: wide|medium|close [on <actor>|<prop>] [slow|quick]
           lamp:   <left|right|centre|front|back> [dim|bright] [slow|quick]
  place    centre | front | back | left | right | front left | front right | back left | back right
           | by <prop> | beside <prop>
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOVES = ["hop", "nod", "shake", "tilt", "spin", "squash", "wobble"]
MOVE_DUR = {"hop": 0.55, "nod": 0.8, "shake": 0.7, "tilt": 0.9, "spin": 1.0, "squash": 0.5, "wobble": 0.9, "bump": 0.4}
SIZE = {"small": 0.6, None: 1.0, "big": 1.5}
PACE = {"quick": 0.6, None: 1.2, "slow": 2.2}
PLACES = {"centre": (0, 0.45), "center": (0, 0.45), "front": (0, 0.28), "back": (0, 0.7), "left": (-0.25, 0.45),
          "right": (0.25, 0.45), "front left": (-0.25, 0.3), "front right": (0.25, 0.3),
          "back left": (-0.25, 0.68), "back right": (0.25, 0.68)}
CAMERA_Z = {"wide": 0.0, "medium": 0.28, "close": 0.55}


# ---------------------------------------------------------------- the stage sheet: what the director may refer to
def set_json(shot):
    """The set's set.json (samples/sets/<name>/set.json next to <name>.set.png, else decoded from the PNG)."""
    url = shot["set"].lstrip("/")
    side = ROOT / url.replace(".set.png", "") / "set.json"
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    import m3ds
    data, _ = m3ds.decode_png_bytes((ROOT / url).read_bytes())
    return json.loads(data)


def prop_anchors(s):
    """Stage (x, z) of each prop's floor anchor -- the same bottom-centre pivot the player uses (setcore.buildSet)."""
    (vx, vy), f, D, bw = s["vp"], s["focal"], s["depth"], s["box"][0]
    C = s["front_rect"]; cx = (C[0][0] + C[1][0]) / 2
    out = []
    for pr in s["props"]:
        bx0, by0, bx1, by1 = pr["bbox"]
        z = pr.get("z", D / 2)
        if pr.get("fold"):
            zc = pr["fold"]["zcols"]; z = zc[len(zc) >> 1]
        elif pr.get("plane"):
            a, b, c = pr["plane"]; xm = (bx0 + bx1) / 2
            z = (c - a * vx + b * f) / (a * (xm - vx) + b * f) * f - f
        X = vx + ((bx0 + bx1) / 2 - vx) * (f + z) / f
        out.append({"x": round((X - cx) / bw, 3), "z": round(z / D, 3), "anchor": pr.get("anchor"), "fold": bool(pr.get("fold"))})
    return out


def stage_sheet(shot):
    """Everything the director sees and the compiler needs: props (named, positioned, movable?), cast, sentences."""
    s = set_json(shot)
    names = shot.get("prop_names") or [f"prop {i}" for i in range(len(s["props"]))]
    movable = set(shot.get("prop_movable", []))
    props = []
    for i, (a, name) in enumerate(zip(prop_anchors(s), names)):
        if name:
            props.append({"index": i, "name": name.lower(), **a, "movable": i in movable})
    cast, sentences = [], []
    for act in shot.get("actors", []):
        if act["kind"] != "puppet": continue
        cast.append(act["id"])
        for ln in act.get("lines", []):
            for sn in line_sentences(ln):
                sentences.append({"actor": act["id"], **sn})
    sentences.sort(key=lambda x: x["t0"])
    for n, sn in enumerate(sentences, 1): sn["n"] = n
    return {"props": props, "cast": cast, "sentences": sentences, "duration": shot["duration"], "set": s}


def line_sentences(ln):
    """Sentences of one shot line in SHOT time, from the direction sheet (`line.sheet`, or <audio base>.sheet.png)."""
    sheet_url = ln.get("sheet") or re.sub(r"\.cloud\.(ogg|mp3|wav)$|\.(ogg|mp3|wav)$", "", ln["audio"]) + ".sheet.png"
    p = ROOT / sheet_url.lstrip("/")
    if not p.exists(): return []
    import sheet as S
    sh = S.loads(p.read_bytes())
    fr, to, t = ln.get("from", 0), ln.get("to", 1e9), ln["t"]
    out = []
    for se in sh["sentences"]:
        if se["t1"] <= fr + 0.05 or se["t0"] >= to - 0.05: continue
        words = [(round(t + w0 - fr, 3), round(t + w1 - fr, 3), w) for w0, w1, w in se.get("words", []) if fr <= w0 < to]
        out.append({"t0": round(t + max(se["t0"], fr) - fr, 3), "t1": round(t + min(se["t1"], to) - fr, 3),
                    "text": se["text"], "delivery": se.get("delivery"), "words": words})
    return out


# ---------------------------------------------------------------- parsing
WHEN = re.compile(r"""^(?:(?P<sec>\d+(?:\.\d+)?)|S(?P<n>\d+)(?:(?P<end>\.end)|:(?:"(?P<qw>[^"]+)"|(?P<w>[\w'’-]+)))?)
                      (?P<off>[+-]\d+(?:\.\d+)?)?$""", re.X | re.I)


class BlockError(ValueError):
    pass


def resolve_when(tok, sheet):
    m = WHEN.match(tok)
    if not m: raise BlockError(f"bad time {tok!r}")
    if m["sec"] is not None:
        t = float(m["sec"])
    else:
        n = int(m["n"])
        if not 1 <= n <= len(sheet["sentences"]): raise BlockError(f"no sentence S{n}")
        sn = sheet["sentences"][n - 1]
        word = m["qw"] or m["w"]
        if m["end"]:
            t = sn["t1"]
        elif word:
            key = re.sub(r"\W", "", word.lower())
            hit = [w for w in sn["words"] if re.sub(r"\W", "", w[2].lower()) == key]
            if not hit: raise BlockError(f"S{n} has no word {word!r}")
            t = hit[0][0]
        else:
            t = sn["t0"]
    return max(0.0, min(sheet["duration"], t + float(m["off"] or 0)))


def split_line(line, sheet):
    """-> (when_token, subject, rest_tokens). Subjects may be multi-word (prop names): longest match wins."""
    parts = line.split(None, 1)
    if len(parts) < 2: raise BlockError("expected '<when> <subject> <verb> ...'")
    when, rest = parts[0], parts[1].strip().lower()
    subjects = sorted(["camera", "lamp"] + [c.lower() for c in sheet["cast"]] + [p["name"] for p in sheet["props"]],
                      key=len, reverse=True)
    for s in subjects:
        if rest == s or rest.startswith(s + " "):
            return when, s, rest[len(s):].split()
    # a phrase anchor ("S3:But what bob ...") -- the beat lands on the phrase's first word; skip the rest of it
    if re.match(r"^S\d+:\w", when, re.I):
        words = rest.split()
        for i in range(1, min(6, len(words))):
            tail = " ".join(words[i:])
            for s in subjects:
                if tail == s or tail.startswith(s + " "):
                    return when, s, tail[len(s):].split()
    raise BlockError(f"unknown subject in {rest!r} (cast: {', '.join(sheet['cast'])}; props: "
                     f"{', '.join(p['name'] for p in sheet['props'])})")


def find_prop(name, sheet):
    for p in sheet["props"]:
        if p["name"] == name: return p
    return None


def parse_target(words, sheet):
    """A prop/actor name from the remaining words (multi-word props)."""
    s = " ".join(words)
    if s in [c.lower() for c in sheet["cast"]]: return ("actor", s)
    p = find_prop(s, sheet)
    if p: return ("prop", p)
    raise BlockError(f"unknown target {s!r}")


# ---------------------------------------------------------------- compiling
class Compiler:
    def __init__(self, shot, sheet):
        self.shot, self.sheet = shot, sheet
        self.pos = {}         # actor -> current (x, z) as blocked so far (used for push directions and face)
        self.errors, self.used = [], []

    def clamp(self, t):
        return round(min(self.sheet["duration"], t), 3)

    def place(self, words, actor):
        s = " ".join(words)
        s = re.sub(r"^(front|back) (centre|center|middle)$", r"\1", s)   # "back centre" = back
        s = {"middle": "centre", "center": "centre"}.get(s, s)
        if s in PLACES: return PLACES[s]
        for kind in ("by", "beside"):
            if s.startswith(kind + " "):
                p = find_prop(s[len(kind) + 1:], self.sheet)
                if not p: raise BlockError(f"no prop {s[len(kind) + 1:]!r}")
                if kind == "by":                             # in front of it (toward the camera); props against a side
                    lim = 0.3 if p["anchor"] in ("left_wall", "right_wall", "folded") else 0.42   # wall: stop short of it
                    return (max(-lim, min(lim, p["x"])), max(0.18, p["z"] - 0.14))
                side = -1 if p["x"] > 0 else 1               # beside it, on the side nearer the room's centre
                return (max(-0.42, min(0.42, p["x"] + side * 0.14)), max(0.18, p["z"] - 0.04))
        raise BlockError(f"unknown place {s!r} (use {', '.join(PLACES)} or by/beside <prop>)")

    def actor(self, aid):
        return next(a for a in self.shot["actors"] if a["id"].lower() == aid)

    def prop_actor(self, p):
        for a in self.shot["actors"]:
            if a["kind"] == "prop" and a["prop"] == p["index"]: return a
        a = {"id": p["name"], "kind": "prop", "prop": p["index"], "keys": [{"t": 0, "dx": 0, "dz": 0, "turn": 0}], "moves": []}
        self.shot["actors"].append(a)
        return a

    def compile(self, script):
        beats = []
        for raw in script.splitlines():
            line = raw.split("#", 1)[0].strip().strip("`").strip()
            line = re.sub(r"^[-*]\s+", "", line)
            if not line: continue
            try:
                when, subj, rest = split_line(line, self.sheet)
                beats.append((round(resolve_when(when, self.sheet), 3), line, subj, rest))
            except BlockError as e:
                self.errors.append(f"{line}  -> {e}")
        for t, line, subj, rest in sorted(beats, key=lambda b: b[0]):   # stable: script order breaks ties
            try:
                if not rest: raise BlockError("missing verb")
                if subj == "camera": self.camera(t, rest)
                elif subj == "lamp": self.lamp(t, rest)
                elif subj in [c.lower() for c in self.sheet["cast"]]: self.puppet(t, subj, rest)
                else: self.prop(t, find_prop(subj, self.sheet), rest)
                self.used.append(line)
            except BlockError as e:
                self.errors.append(f"{line}  -> {e}")
        for a in self.shot["actors"]:
            a["keys"] = merge_keys(a.get("keys", []))
            a.setdefault("moves", []).sort(key=lambda m: m["t"])
        self.shot["camera"] = merge_keys(self.shot["camera"])
        self.shot["lamp"] = merge_keys(self.shot["lamp"])
        self.shot["blocking"] = self.used
        return self.shot

    def puppet(self, t, aid, rest):
        a, verb, args = self.actor(aid), rest[0], rest[1:]
        if verb == "at":
            x, z = self.place(args, a)
            if t > 0: interrupt(a["keys"], ("x", "z"), t)
            a["keys"].append({"t": t, "x": x, "z": z, "ease": "hold"}); self.pos[aid] = (x, z)
        elif verb == "go":
            pace = args[-1] if args and args[-1] in PACE else None
            x, z = self.place(args[:-1] if pace else args, a)
            interrupt(a["keys"], ("x", "z"), t)                  # leave from wherever it is at t
            a["keys"].append({"t": self.clamp(t + PACE[pace]), "x": x, "z": z, "ease": "smooth"}); self.pos[aid] = (x, z)
        elif verb == "face":
            tgt = args
            if tgt == ["camera"]: turn = 0
            elif tgt in (["left"], ["right"]): turn = -35 if tgt == ["left"] else 35
            else:
                kind, obj = parse_target(tgt, self.sheet)
                ox = obj["x"] if kind == "prop" else self.pos.get(obj, (0, 0))[0]
                sx = self.pos.get(aid, (0, 0))[0]
                turn = 30 if ox > sx else -30
            interrupt(a["keys"], ("turn",), t)
            a["keys"].append({"t": self.clamp(t + 0.4), "turn": turn, "ease": "smooth"})
        elif verb in ("push", "nudge", "shove", "bump", "kick"):   # "<actor> push <prop> [dir]" = the prop's push/bump
            words = [w for w in args if w not in SIZE and w not in ("away", "back", "forward", "left", "right")] or args
            kind, p = parse_target(words, self.sheet)
            if kind != "prop": raise BlockError(f"{aid} can only push props")
            extra = [w for w in args if w not in words]
            if verb == "bump" or not p["movable"]: self.prop(t, p, ["bump"] + extra)
            else: self.prop(t, p, ["push"] + (extra or ["away"]) + (["small"] if verb == "nudge" else []))
        elif verb in MOVES:
            size = args[0] if args and args[0] in SIZE else None
            a.setdefault("moves", []).append({"t": t, "move": verb, "dur": MOVE_DUR[verb], "amp": SIZE[size]})
        else:
            raise BlockError(f"unknown actor verb {verb!r}")

    def prop(self, t, p, rest):
        verb, args = rest[0], rest[1:]
        size = next((w for w in args if w in SIZE), None)
        if verb == "bump":
            self.prop_actor(p)["moves"].append({"t": t, "move": "bump", "dur": MOVE_DUR["bump"], "amp": SIZE[size]})
            return
        if verb != "push": raise BlockError(f"unknown prop verb {verb!r}")
        if not p["movable"]: raise BlockError(f"{p['name']} is fixed; only bump it")
        d = next((w for w in args if w in ("away", "back", "forward", "left", "right")), "away")
        k = 0.08 * SIZE[size]
        if d == "away":                                         # away from the nearest actor
            near = min(self.pos.items(), key=lambda kv: abs(kv[1][0] - p["x"]) + abs(kv[1][1] - p["z"]), default=None)
            ax, az = near[1] if near else (p["x"], p["z"] - 0.2)
            vx, vz = p["x"] - ax, p["z"] - az
            n = max(1e-6, (vx * vx + vz * vz) ** 0.5); dx, dz = k * vx / n, k * vz / n
        else:
            dx, dz = {"back": (0, k), "forward": (0, -k), "left": (-k, 0), "right": (k, 0)}[d]
        a = self.prop_actor(p)
        cur = {c: v or 0 for c, v in interrupt(a["keys"], ("dx", "dz", "turn"), t).items()}
        a["keys"].append({"t": self.clamp(t + 0.8), "dx": round(cur["dx"] + dx, 3), "dz": round(cur["dz"] + dz, 3),
                          "turn": round(cur["turn"] + (14 if dx >= 0 else -14) * SIZE[size], 1), "ease": "out"})

    def camera(self, t, rest):
        shot, verb = rest[0], rest[1:]
        if shot not in CAMERA_Z: raise BlockError(f"camera wants wide|medium|close, got {shot!r}")
        pace = verb[-1] if verb and verb[-1] in PACE else None
        if pace: verb = verb[:-1]
        x = y = 0.0
        if verb and verb[0] == "on":
            kind, obj = parse_target(verb[1:], self.sheet)
            tx = obj["x"] if kind == "prop" else self.pos.get(obj, (0, 0.45))[0]
            x = max(-1, min(1, round(tx * 3.5 * (0.4 + CAMERA_Z[shot]), 3)))   # lean toward the subject, more when close
            y = 0.1 if kind == "actor" and shot != "wide" else 0.0
        cam = self.shot["camera"]
        dur = PACE[pace] * 2.5 if t > 0 else 0                      # camera moves are slower than people
        if dur: interrupt(cam, ("x", "y", "z"), t)
        cam.append({"t": self.clamp(t + dur), "x": x, "y": y, "z": CAMERA_Z[shot], "ease": "smooth"})

    def lamp(self, t, rest):
        pos = {"left": (-0.3, 0.45), "right": (0.3, 0.45), "centre": (0, 0.45), "center": (0, 0.45),
               "front": (0, 0.2), "back": (0, 0.8)}
        if rest[0] not in pos: raise BlockError(f"lamp wants left|right|centre|front|back, got {rest[0]!r}")
        li = 0.7 if "dim" in rest else 1.6 if "bright" in rest else 1.2
        pace = next((w for w in rest if w in PACE), None)
        lamp = self.shot["lamp"]
        if t > 0: interrupt(lamp, ("lx", "ly", "lz", "li"), t)
        lx, lz = pos[rest[0]]
        lamp.append({"t": self.clamp(t + (PACE[pace] * 2 if t > 0 else 0)), "lx": lx, "ly": -0.33, "lz": lz, "li": li, "ease": "smooth"})


def sample(keys, ch, t):
    """Value of a channel at t (linear between keys; eases ignored -- only used to start a new move from mid-flight)."""
    ks = [k for k in sorted(keys, key=lambda k: k["t"]) if ch in k]
    if not ks: return None
    if t <= ks[0]["t"]: return ks[0][ch]
    for a, b in zip(ks, ks[1:]):
        if a["t"] <= t <= b["t"]:
            u = (t - a["t"]) / max(1e-9, b["t"] - a["t"])
            return round(a[ch] + (b[ch] - a[ch]) * u, 4) if isinstance(a[ch], (int, float)) else a[ch]
    return ks[-1][ch]


def interrupt(keys, chans, t):
    """A new move on `chans` starting at t supersedes anything planned after t: drop those channels from later keys and
    pin their current (mid-flight) values at t. Returns the pinned values."""
    now = {c: sample(keys, c, t) for c in chans}
    for k in keys:
        if k["t"] > t + 1e-6:
            for c in chans: k.pop(c, None)
    keys[:] = [k for k in keys if k["t"] <= t + 1e-6 or set(k) - {"t", "ease"}]
    pinned = {c: v for c, v in now.items() if v is not None}
    if pinned: keys.append({"t": t, **pinned})
    return now


def last_value(keys, ch, t):
    v = None
    for k in sorted(keys, key=lambda k: k["t"]):
        if k["t"] <= t + 1e-6 and ch in k: v = k[ch]
    return v


def merge_keys(keys):
    """Sort; merge keys at the same time (later wins per channel)."""
    out = {}
    for k in sorted(keys, key=lambda k: k["t"]):
        out.setdefault(round(k["t"], 3), {}).update(k)
    return [out[t] for t in sorted(out)]


def blank_shot(shot):
    """The shot minus its blocking: keeps set, duration, lines, sfx/music/markers and puppet looks; drops keys/moves,
    camera and lamp, and props' actors (the director re-adds the props it moves)."""
    s = json.loads(json.dumps(shot))
    s["actors"] = [a for a in s["actors"] if a["kind"] != "prop"]
    for a in s["actors"]:
        k0 = a.get("keys", [{}])[0]
        keep = {c: k0[c] for c in ("lift", "height") if c in k0}
        a["keys"] = [{"t": 0, "x": 0, "z": 0.45, "turn": 0, "lift": 0.24 if a.get("float") else 0, "height": 0.36, **keep}]
        a["moves"] = []
    s["camera"] = [{"t": 0, "x": 0, "y": 0, "z": 0}]
    s["lamp"] = [{"t": 0, "lx": -0.15, "ly": -0.3, "lz": 0.45, "li": 1.2}]
    s.pop("blocking", None)
    return s


def portable_stage(shot):
    """The stage sheet without the set geometry: everything the compiler and the director need, small enough to
    travel inside the shot package (so a Graydient node can compile without the set files or direction sheets)."""
    st = stage_sheet(blank_shot(shot))
    st.pop("set", None)
    return st


def stage_card(stage, brief=""):
    """The plain-text stage card an LLM (Graydient skill / local director) blocks from."""
    def where(p):
        lr = "left" if p["x"] < -0.2 else "right" if p["x"] > 0.2 else "centre"
        fb = "front" if p["z"] < 0.35 else "back" if p["z"] > 0.65 else "middle"
        return f"{fb} {lr}"
    props = "\n".join(f"- {p['name']} ({where(p)}, {'movable' if p['movable'] else 'fixed'})" for p in stage["props"])
    lines = "\n".join(f"S{s['n']} ({s['actor']}, {s['t0']:.1f}-{s['t1']:.1f} s, {s.get('delivery') or 'neutral'}): {s['text']}"
                      for s in stage["sentences"])
    return (f"SCENE: {stage['duration']:.0f} seconds.\n" + (f"BRIEF: {brief}\n" if brief else "")
            + f"ACTORS: {', '.join(stage['cast'])} (floating cartoon heads)\nPROPS:\n{props}\nDIALOGUE:\n{lines}\n")


def compile_script(shot, script, stage=None):
    """-> (compiled shot, errors). `shot` supplies cast/lines; its existing blocking is replaced. `stage` (from
    portable_stage / a shot package) replaces the lookup of set.json and direction sheets."""
    base = blank_shot(shot)
    sheet = json.loads(json.dumps(stage)) if stage else stage_sheet(base)
    c = Compiler(base, sheet)
    return c.compile(script), c.errors
