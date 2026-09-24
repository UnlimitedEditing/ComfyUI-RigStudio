"""Audio -> expression track (docs/expression-track.md, format v0).

    python analyze.py clip.wav -o clip.track.json
    python analyze.py other_speaker.wav --role listener -o listener.track.json

Everything here is CPU, runs once per clip. Layers produced:
  idle    blinks, saccades, gaze aversion, breathing, head drift, mood baseline
  reflex  prosody-driven: pitch accents -> brows, stressed syllables -> nods,
          rising phrase ends -> question tilt, speech energy -> head motion.
          Listener role: backchannel nods/smiles/brow flicks at the other
          speaker's phrase ends and accents instead.
Visemes come from Rhubarb Lip Sync when it is available (--rhubarb or on PATH),
otherwise from a crude energy/spectrum heuristic that only exists to keep the
pipeline runnable.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

# Gesture timing measured on CREMA-D (1621 clips, 20 actors; analysis/crema_stats.py):
BROW_LEAD = 0.25      # brow peak precedes the pitch accent by ~250 ms
NOD_LEAD = 0.35       # accent-linked nod peaks ~400 ms before the accent
# pulse() uses smoothstep edges, whose 10-90% span is ~0.6 of the nominal time, so the
# constants are the measured 10-90% times / 0.6 (validated with score.py: ratio ~1.0)
BROW_RISE, BROW_FALL = 0.167 / 0.6, 0.267 / 0.6
NOD_RISE, NOD_FALL = 0.167 / 0.6, 0.2 / 0.6
BLINK_CLOSE, BLINK_OPEN = 0.033, 0.167 / 0.6
BROW_LR_SPREAD = 0.4  # real outer brows differ L/R by ~40% on average (corr 0.91)

SR = 16000
HOP = 160                      # 10 ms analysis frames
AFPS = SR / HOP                # analysis frames per second
FPS = 30                       # output curve rate


# ---------------------------------------------------------------- features

def analyse_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-6)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=450, sr=SR, frame_length=1024,
                                 hop_length=HOP)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=640, hop_length=HOP)[0]
    n = min(len(db), len(f0), len(cent))
    db, f0, voiced, cent = db[:n], f0[:n], voiced[:n], cent[:n]

    floor = np.percentile(db, 10)
    peak = np.percentile(db, 98)
    speech = db > max(floor + 12, peak - 35)
    speech = median_filter(speech.astype(np.uint8), size=7).astype(bool)

    # f0 in semitones relative to the speaker's median, NaN where unvoiced
    st = np.full(n, np.nan)
    ok = voiced & np.isfinite(f0) & speech
    if ok.any():
        st[ok] = 12 * np.log2(f0[ok] / np.median(f0[ok]))
        st_s = median_filter(np.where(ok, st, 0.0), size=5)
        st[ok] = st_s[ok]
    return dict(y=y, n=n, db=db, st=st, voiced=ok, cent=cent, speech=speech,
                duration=len(y) / SR)


def phrases_from(speech, min_pause=0.25, min_len=0.3):
    """Speech runs separated by pauses >= min_pause seconds -> [(t0, t1)]."""
    runs, start = [], None
    for i, s in enumerate(np.append(speech, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append([start, i]); start = None
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) / AFPS < min_pause:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a / AFPS, b / AFPS) for a, b in merged if (b - a) / AFPS >= min_len]


def pitch_accents(a):
    """Local f0 peaks -> [(t, strength 0..1)], at most one per ~0.9 s."""
    st = a["st"]
    ok = np.isfinite(st)
    if ok.sum() < 10:
        return []
    idx = np.arange(len(st))
    filled = gaussian_filter1d(np.interp(idx, idx[ok], st[ok]), 2)   # bridge unvoiced gaps
    peaks, props = find_peaks(filled, prominence=2.0, distance=int(0.3 * AFPS),
                              wlen=int(1.2 * AFPS))
    speech_db = np.median(a["db"][a["speech"]]) if a["speech"].any() else 0
    cands = []
    for p, prom in zip(peaks, props["prominences"]):
        if not ok[p] or a["db"][p] < speech_db - 6:
            continue
        cands.append((p / AFPS, float(np.clip((prom - 1.5) / 6.0, 0.15, 1.0))))
    out = []                                   # strongest first, enforce spacing
    for t, s in sorted(cands, key=lambda c: -c[1]):
        if all(abs(t - u) >= 0.9 for u, _ in out):
            out.append((t, s))
    return sorted(out)


def stresses(a):
    env = gaussian_filter1d(a["db"], 3)
    sp = a["speech"]
    if not sp.any():
        return []
    med, sd = np.median(env[sp]), np.std(env[sp]) + 1e-6
    peaks, _ = find_peaks(env, prominence=4.0, distance=int(0.35 * AFPS))
    out = []
    for p in peaks:
        z = (env[p] - med) / sd
        if sp[p] and z > 0.6:
            out.append((p / AFPS, float(np.clip((z - 0.6) / 1.5, 0.15, 1.0))))
    return out


def phrase_ending(a, t0, t1):
    """Semitone rise over the phrase's final voiced stretch (+ = rising)."""
    i0, i1 = int(t0 * AFPS), int(t1 * AFPS)
    st = a["st"][i0:i1]
    v = st[np.isfinite(st)]
    if len(v) < 12:
        return 0.0
    q = max(3, len(v) // 5)
    return float(np.mean(v[-q:]) - np.mean(v[-2 * q:-q]))


# ---------------------------------------------------------------- curve ops

class Layer:
    def __init__(self, frames):
        self.frames = frames
        self.curves = {}

    def c(self, name):
        if name not in self.curves:
            self.curves[name] = np.zeros(self.frames)
        return self.curves[name]

    def pulse(self, name, t, amp, attack=0.08, hold=0.15, decay=0.4):
        """Attack/hold/decay envelope with smoothstep edges."""
        c = self.c(name)
        tt = np.arange(self.frames) / FPS - t
        env = np.zeros(self.frames)
        up = (tt >= -attack) & (tt < 0)
        env[up] = _smooth((tt[up] + attack) / attack)
        env[(tt >= 0) & (tt < hold)] = 1
        dn = (tt >= hold) & (tt < hold + decay)
        env[dn] = 1 - _smooth((tt[dn] - hold) / decay)
        c += amp * env

    def nod(self, t, deg):
        """Biphasic head nod: dip down, return with a small overshoot."""
        self.pulse("headPitch", t, deg, attack=NOD_RISE, hold=0.04, decay=NOD_FALL)
        self.pulse("headPitch", t + 0.32, -0.25 * deg, attack=0.12, hold=0.0, decay=0.25)

    def blink(self, t):
        for s in ("eyeBlink_L", "eyeBlink_R"):
            self.pulse(s, t, 1.0, attack=BLINK_CLOSE, hold=0.03, decay=BLINK_OPEN)

    def brows(self, t, inner, outer, rng):
        """Brow raise with real rise/fall and independent L/R amplitude + tiny L/R offset."""
        self.pulse("browInnerUp", t, inner, attack=BROW_RISE, hold=0.06, decay=BROW_FALL)
        for side in ("L", "R"):
            k = 1 + rng.uniform(-BROW_LR_SPREAD, BROW_LR_SPREAD) / 2
            self.pulse(f"browOuterUp_{side}", t + rng.uniform(-0.03, 0.03), outer * k,
                       attack=BROW_RISE, hold=0.06, decay=BROW_FALL)

    def export(self, weight=1.0, blend="add"):
        out = {}
        for k, v in self.curves.items():
            nd = 2 if k.startswith(("head",)) else 3
            out[k] = [round(float(x), nd) for x in v]
        return {"blend": blend, "weight": weight, "curves": out}


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def smooth_noise(rng, frames, sigma_s, amp):
    raw = rng.standard_normal(frames + int(6 * sigma_s * FPS))
    f = gaussian_filter1d(raw, sigma_s * FPS)[: frames]
    return amp * f / (np.std(f) + 1e-9)


def gaze_track(rng, frames, fixations):
    """fixations: sorted [(t, x, y)] -> step curves with 40 ms saccades."""
    t = np.arange(frames) / FPS
    gx = np.full(frames, fixations[0][1])
    gy = np.full(frames, fixations[0][2])
    for ts, x, y in fixations[1:]:
        w = _smooth((t - ts) / 0.04)
        gx = gx * (1 - w) + x * w
        gy = gy * (1 - w) + y * w
    return gx, gy


# ---------------------------------------------------------------- layers

def build_idle(a, phrases, frames, rng, role, mood, events, idle_face=True):
    """idle_face=False drops the continuous low-level brow activity — used when an A2F base
    layer already supplies continuous emotional expression (CREMA-D: brows 1.96x -> 1.37x real;
    the idle squint is kept, A2F under-drives squint)."""
    L = Layer(frames)
    dur = frames / FPS
    t = np.arange(frames) / FPS

    # head drift + breathing
    L.c("headYaw")[:] += smooth_noise(rng, frames, 1.2, 2.2)
    L.c("headPitch")[:] += smooth_noise(rng, frames, 1.4, 1.2) + 0.5 * np.sin(2 * np.pi * 0.22 * t)
    L.c("headRoll")[:] += smooth_noise(rng, frames, 1.6, 0.9)
    if role == "speaker":
        for t0, _ in phrases:          # inhale lift just before speaking
            L.pulse("headPitch", t0 - 0.15, -1.0, attack=0.25, hold=0.05, decay=0.5)

    # gaze: fixations with micro-saccades, aversion at phrase starts
    fix = [(0.0, 0.0, 0.0)]
    big_shifts = []
    avert_p = 0.45 if role == "speaker" else 0.12
    aversions = []
    for t0, t1 in phrases:
        if t1 - t0 >= 1.2 and rng.random() < avert_p:
            d = rng.uniform(0.5, 1.4)
            x = rng.choice([-1, 1]) * rng.uniform(0.35, 0.7)
            y = rng.uniform(-0.25, 0.4)
            aversions.append((t0 + rng.uniform(-0.1, 0.25), d, x, y))
    ts, cx, cy = 0.0, 0.0, 0.0
    ai = 0
    while ts < dur:
        nxt = ts + rng.uniform(0.35, 1.1)
        if ai < len(aversions) and nxt >= aversions[ai][0]:
            at, d, x, y = aversions[ai]
            at = max(at, ts + 0.05)
            fix.append((at, x, y)); big_shifts.append(at)
            events.append({"t": round(at, 3), "kind": "gaze_aversion", "dur": round(d, 2)})
            ts = at + d
            cx, cy = rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05)
            fix.append((ts, cx, cy)); big_shifts.append(ts)
            ai += 1
            continue
        ts = nxt
        fix.append((ts, cx + rng.uniform(-0.06, 0.06), cy + rng.uniform(-0.04, 0.04)))
    gx, gy = gaze_track(rng, frames, sorted(fix))
    L.c("gazeX")[:] = gx
    L.c("gazeY")[:] = gy

    # blinks: at speech onset (CREMA-D: clear blink spike exactly at onset), rarely at
    # pause starts, on big gaze shifts, then fill gaps
    cands = []
    for t0, t1 in phrases:
        if rng.random() < 0.35:
            cands.append(t0 + rng.uniform(-0.05, 0.05))
        if rng.random() < 0.15:
            cands.append(t1 + rng.uniform(0.05, 0.15))
    for s in big_shifts:
        if rng.random() < 0.5:
            cands.append(s)
    blinks = []
    for c in sorted(cands):
        if 0 < c < dur and (not blinks or c - blinks[-1] > 0.9):
            blinks.append(c)
    filled, last = [], -rng.uniform(0.5, 2.0)
    for b in blinks + [dur + 10]:
        while b - last > 5.5:
            last += rng.uniform(2.5, 5.0)
            if last < dur and (b - last > 0.9):
                filled.append(last)
        if b < dur:
            filled.append(b); last = b
    for b in filled:
        L.blink(b)
        events.append({"t": round(b, 3), "kind": "blink"})
        if rng.random() < 0.12:
            L.blink(b + 0.32)

    # low-level continuous activity: real brows/lids are almost never exactly at rest
    _idle_face(L, rng, frames, brows=idle_face)

    if mood:
        side = ("mouthSmile_L", "mouthSmile_R") if mood > 0 else ("mouthFrown_L", "mouthFrown_R")
        for s in side:
            L.c(s)[:] += abs(mood) * 0.35
    return L


def _idle_face(L, rng, frames, brows=True):
    if brows:
        L.c("browInnerUp")[:] += np.clip(smooth_noise(rng, frames, 0.5, 0.04) + 0.03, 0, None)
    for side in ("L", "R"):
        if brows:
            L.c(f"browOuterUp_{side}")[:] += np.clip(smooth_noise(rng, frames, 0.6, 0.03) + 0.02, 0, None)
        L.c(f"eyeSquint_{side}")[:] += np.clip(smooth_noise(rng, frames, 0.8, 0.03) + 0.03, 0, None)


WH = __import__("re").compile(r"^(?:\W*(?:well|so|and|but|okay|ok|now|oh)\W+)*"
                              r"(what|why|how|where|who|whom|which|when)\b", __import__("re").I)


def build_reflex_speaker(a, phrases, frames, rng, events, words=None):
    L = Layer(frames)
    acc = pitch_accents(a)
    strs = stresses(a)

    # speech energy -> head motion amplitude
    env = np.interp(np.arange(frames) / FPS, np.arange(a["n"]) / AFPS,
                    np.clip((a["db"] - np.percentile(a["db"], 30)) / 25, 0, 1))
    env = gaussian_filter1d(env, 0.3 * FPS)
    L.c("headYaw")[:] += env * smooth_noise(rng, frames, 0.35, 2.2)
    L.c("headPitch")[:] += env * smooth_noise(rng, frames, 0.3, 1.8)
    L.c("headRoll")[:] += env * smooth_noise(rng, frames, 0.4, 1.4)

    acc_t = np.array([t for t, _ in acc]) if acc else np.array([])
    for t, s in acc:
        emph = any(abs(t - st) < 0.12 for st, _ in strs)
        # brows peak BROW_LEAD before every accent, emphasised or not (CREMA-D)
        tb = t - BROW_LEAD
        outer = (0.35 + 0.4 * s) if emph else 0.2 * s
        L.brows(tb, 0.25 + 0.35 * s, outer, rng)
        if emph and s > 0.5:
            for e in ("eyeWide_L", "eyeWide_R"):
                L.pulse(e, tb, 0.3 * s, attack=BROW_RISE, hold=0.06, decay=BROW_FALL)
        events.append({"t": round(t, 3), "kind": "emphasis" if emph else "pitch_accent",
                       "strength": round(s, 2)})

    # nods: strongest stresses first (accent-aligned ones score higher), >= 1.6 s apart
    scored = []
    for t, s in strs:
        near_acc = bool(len(acc_t)) and np.min(np.abs(acc_t - t)) < 0.12
        scored.append((s + (0.4 if near_acc else 0.0) + rng.uniform(0, 0.1), t, s))
    nods = []
    for score, t, s in sorted(scored, reverse=True):
        if score >= 0.4 and all(abs(t - u) >= 1.6 for u in nods):
            nods.append(t)
            near_acc = bool(len(acc_t)) and np.min(np.abs(acc_t - t)) < 0.12
            L.nod(t - (NOD_LEAD if near_acc else 0.05), 2.0 + 3.0 * min(1.0, score))
            events.append({"t": round(t, 3), "kind": "stress_nod", "strength": round(s, 2)})

    def question(t1, wh):
        hold, side = 0.6, rng.choice([-1, 1])
        if wh:      # wh-questions: slight furrow + tilt (brows raise on yes/no questions)
            for s in ("browDown_L", "browDown_R"):
                L.pulse(s, t1 - 0.4, 0.3, attack=0.25, hold=hold, decay=0.5)
            L.pulse("browInnerUp", t1 - 0.2, 0.15, attack=0.2, hold=hold, decay=0.5)
        else:
            L.pulse("browInnerUp", t1 - 0.25, 0.35, attack=0.2, hold=hold, decay=0.5)
            L.pulse("browOuterUp_L", t1 - 0.25, 0.3, attack=0.2, hold=hold, decay=0.5)
            L.pulse("browOuterUp_R", t1 - 0.25, 0.3, attack=0.2, hold=hold, decay=0.5)
        L.pulse("headRoll", t1 - 0.3, side * 4.0, attack=0.35, hold=hold, decay=0.6)

    for t0, t1 in phrases:
        events.append({"t": round(t0, 3), "kind": "phrase", "end": round(t1, 3)})
        rise = phrase_ending(a, t0, t1)
        if words is None and rise > 2.5:       # no transcript: rising end ~ question
            question(t1, wh=False)
            events.append({"t": round(t1, 3), "kind": "question", "rise": round(rise, 1)})
        elif rise < -2.0:                      # falling, declarative close
            L.nod(t1 - 0.1, 1.5)
    for s in (words or {}).get("sentences", []):
        if s["text"].endswith("?"):
            wh = bool(WH.search(s["text"]))
            question(s["t1"], wh)
            events.append({"t": round(s["t1"], 3), "kind": "question",
                           "wh": wh, "text": s["text"][-60:]})
    for k in ("browOuterUp_L", "browOuterUp_R", "browInnerUp", "eyeWide_L", "eyeWide_R"):
        if k in L.curves:
            np.clip(L.curves[k], 0, 1, out=L.curves[k])
    return L


def build_reflex_listener(a, phrases, frames, rng, events):
    """React to the OTHER speaker's audio: backchannel at their phrase ends."""
    L = Layer(frames)
    for i, (t0, t1) in enumerate(phrases):
        pause = (phrases[i + 1][0] if i + 1 < len(phrases) else t1 + 2.0) - t1
        if t1 - t0 < 1.0 or pause < 0.35:
            continue
        r = rng.random()
        if r < 0.55:
            n = 1 if rng.random() < 0.5 else 2
            for k in range(n):
                L.nod(t1 + 0.15 + 0.42 * k, rng.uniform(2.0, 3.5))
            events.append({"t": round(t1, 3), "kind": "backchannel_nod", "count": n})
        elif r < 0.8:
            for s in ("mouthSmile_L", "mouthSmile_R"):
                L.pulse(s, t1 + 0.1, 0.35, attack=0.3, hold=0.6, decay=0.8)
            events.append({"t": round(t1, 3), "kind": "backchannel_smile"})
        if phrase_ending(a, t0, t1) > 2.5:
            L.pulse("browInnerUp", t1, 0.25, attack=0.2, hold=0.5, decay=0.5)
            L.pulse("headRoll", t1, rng.choice([-1, 1]) * 3.0, attack=0.35, hold=0.5, decay=0.6)
            events.append({"t": round(t1, 3), "kind": "attend_question"})
    for t, s in pitch_accents(a):
        if s > 0.6 and rng.random() < 0.35:
            for b in ("browOuterUp_L", "browOuterUp_R"):
                L.pulse(b, t + 0.2, 0.25 * s, attack=0.1, hold=0.1, decay=0.4)
            events.append({"t": round(t + 0.2, 3), "kind": "listen_brow_flick"})
    return L


# ---------------------------------------------------------------- visemes

def _bundled_rhubarb():
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    hits = glob.glob(os.path.join(here, "..", "tools", "Rhubarb-*", "rhubarb*"))
    return next((h for h in hits if h.endswith(("rhubarb", "rhubarb.exe"))), None)


def rhubarb_visemes(path, exe, dialog_text=None):
    """Rhubarb is ~0.5x realtime, so cues are cached next to the audio (keyed on the dialog)."""
    import hashlib
    key = hashlib.sha1((dialog_text or "").encode()).hexdigest()[:10]
    cache = os.path.splitext(path)[0] + f".rhubarb-{key}.json"
    if os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        with open(cache) as f:
            return json.load(f)
    with tempfile.TemporaryDirectory() as td:
        src = path
        if not path.lower().endswith(".wav"):      # Rhubarb can't read opus-in-ogg
            src = os.path.join(td, "in.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", src], check=True)
        out = os.path.join(td, "cues.json")
        cmd = [exe, "-f", "json", "-o", out, "--extendedShapes", "GHX", "-q", src]
        if dialog_text:
            dpath = os.path.join(td, "dialog.txt")
            with open(dpath, "w", encoding="utf-8") as f:
                f.write(dialog_text)
            cmd += ["-d", dpath]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out) as f:
            cues = json.load(f)["mouthCues"]
    cues = [{"t": round(c["start"], 3), "shape": c["value"]} for c in cues]
    with open(cache, "w") as f:
        json.dump(cues, f)
    return cues


def heuristic_visemes(a):
    """Placeholder until Rhubarb is wired in: loudness -> openness, low centroid -> rounded."""
    db, sp, v, cent = a["db"], a["speech"], a["voiced"], a["cent"]
    if not sp.any():
        return [{"t": 0.0, "shape": "X"}]
    p30, p55, p80 = np.percentile(db[sp], [30, 55, 80])
    vc = cent[v] if v.any() else cent[sp]
    c_low = np.percentile(vc, 25)
    shapes = []
    for i in range(a["n"]):
        if not sp[i]:
            s = "X"
        elif db[i] < p30:
            s = "A"
        elif not v[i]:
            s = "B"
        elif cent[i] < c_low and db[i] > p55:
            s = "E" if db[i] > p80 else "F"
        elif db[i] > p80:
            s = "D"
        elif db[i] > p55:
            s = "C"
        else:
            s = "B"
        shapes.append(s)
    # merge runs shorter than 60 ms into the previous shape
    cues, min_len = [], int(0.06 * AFPS)
    i = 0
    while i < len(shapes):
        j = i
        while j < len(shapes) and shapes[j] == shapes[i]:
            j += 1
        if cues and j - i < min_len:
            pass
        elif not cues or cues[-1]["shape"] != shapes[i]:
            cues.append({"t": round(i / AFPS, 3), "shape": shapes[i]})
        i = j
    return cues


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("-o", "--out")
    ap.add_argument("--role", choices=["speaker", "listener"], default="speaker",
                    help="listener: audio is the OTHER speaker; produce reactions, closed mouth")
    ap.add_argument("--mood", type=float, default=0.0, help="-1..1 baseline frown/smile")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rhubarb", help="path to rhubarb executable (else PATH, else heuristic)")
    ap.add_argument("--words", help="transcribe.py output (default: <audio>.words.json if present)")
    args = ap.parse_args()
    wpath = args.words or os.path.splitext(args.audio)[0] + ".words.json"
    words = None
    if os.path.isfile(wpath):
        with open(wpath, encoding="utf-8") as f:
            words = json.load(f)

    rng = np.random.default_rng(args.seed)
    a = analyse_audio(args.audio)
    frames = int(np.ceil(a["duration"] * FPS))
    phrases = phrases_from(a["speech"])
    events = []

    idle = build_idle(a, phrases, frames, rng, args.role, args.mood, events)
    if args.role == "speaker":
        reflex = build_reflex_speaker(a, phrases, frames, rng, events, words)
        exe = args.rhubarb or shutil.which("rhubarb") or _bundled_rhubarb()
        if exe:
            visemes = rhubarb_visemes(args.audio, exe, words["text"] if words else None)
            vsrc = "rhubarb+dialog" if words else "rhubarb"
        else:
            visemes, vsrc = heuristic_visemes(a), "heuristic"
    else:
        reflex = build_reflex_listener(a, phrases, frames, rng, events)
        visemes, vsrc = [{"t": 0.0, "shape": "X"}], "listener"

    track = {
        "format": "rigstudio.expression_track",
        "version": 0,
        "fps": FPS,
        "frames": frames,
        "duration": round(a["duration"], 3),
        "role": args.role,
        "source": {"audio": os.path.basename(args.audio), "generator": "analysis/analyze.py",
                   "visemes": vsrc, "seed": args.seed,
                   "words": os.path.basename(wpath) if words else None},
        "layers": {"idle": idle.export(), "reflex": reflex.export()},
        "visemes": visemes,
        "events": sorted(events, key=lambda e: e["t"]),
    }
    out = args.out or os.path.splitext(args.audio)[0] + ".track.json"
    with open(out, "w") as f:
        json.dump(track, f, separators=(",", ":"))
    kinds = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"{out}: {a['duration']:.1f}s, {len(phrases)} phrases, visemes={vsrc} ({len(visemes)} cues), "
          + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))


if __name__ == "__main__":
    sys.exit(main())
