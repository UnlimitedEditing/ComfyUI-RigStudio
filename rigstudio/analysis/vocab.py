"""Rig Studio direction vocabulary: the single source for the director, the scorer, the direction
sheet and Sheet Studio (the sheet embeds a copy, so Studio needs no hard-coded lists).

Bump VOCAB_VERSION whenever a delivery, function, table row or family changes: sheets record it and
re-directing an older sheet warns instead of silently remapping.
"""
import hashlib
import json

VOCAB_VERSION = 1

VOCAB = {
    "neutral": "plain, informative delivery; no particular colouring",
    "warm": "kind, friendly, inviting",
    "playful": "light fun, a smile in the voice",
    "wry": "dry, knowing irony",
    "teasing": "poking fun, cheeky",
    "excited": "energy up, enthusiastic",
    "amazed": "wonder, genuine surprise",
    "delighted": "relish, sensory pleasure",
    "earnest": "sincere, making a real point",
    "confident": "assured, certain, 'trust me'",
    "proud": "satisfied, showing off a result",
    "skeptical": "doubting, unconvinced",
    "concerned": "worried, cautious",
    "sympathetic": "soft, understanding the listener's difficulty",
    "deadpan": "deliberately flat for comic effect",
}
INTENSITY = ["slight", "clear", "strong"]

FUNCTIONS = {
    "greeting": "hello / welcome / opening line to the viewer",
    "pitch": "selling or praising something, enthusiastic claim",
    "caveat": "hedging, a limitation, a 'but be aware' point",
    "fact": "plain explanation or information",
    "aside": "ironic or knowing side remark, commenting on itself",
    "joke": "a joke, exaggeration or cheeky remark",
    "question": "a question put to the viewer (often rhetorical)",
    "interjection": "a short exclamation or one-word beat (e.g. 'See?', 'Boom!')",
    "heading": "announcing the next section or topic",
    "instruction": "a step to perform",
    "list": "listing items or ingredients",
    "payoff": "the satisfying moment or sensory high point",
    "boast": "showing off a result, 'I told you so'",
    "recommendation": "recommending or pointing the viewer to something good",
    "reassurance": "encouraging the viewer, 'you can do this'",
    "signoff": "closing line / goodbye",
}
# function -> (delivery, intensity).  Written from acting sense, not fitted to the benchmark.
TABLE = {
    "greeting": ("warm", "clear"), "pitch": ("excited", "clear"), "caveat": ("earnest", "slight"),
    "fact": ("neutral", None), "aside": ("wry", "clear"), "joke": ("playful", "clear"),
    "question": ("playful", "clear"), "interjection": ("playful", "clear"), "heading": ("excited", "slight"),
    "instruction": ("neutral", None), "list": ("neutral", None), "payoff": ("delighted", "strong"),
    "boast": ("confident", "clear"), "recommendation": ("warm", "clear"), "reassurance": ("warm", "strong"),
    "signoff": ("warm", "clear"),
}

FAMILY = {"warm": "warm", "sympathetic": "warm", "playful": "fun", "teasing": "fun", "wry": "fun",
          "deadpan": "fun", "excited": "up", "amazed": "up", "delighted": "up", "earnest": "firm",
          "confident": "firm", "proud": "firm", "skeptical": "doubt", "concerned": "doubt", "neutral": "neutral"}
STEP = {"slight": 0, "clear": 1, "strong": 2}


def vocabulary():
    """JSON-ready copy of everything above plus a content hash (catches unbumped edits)."""
    body = {"deliveries": VOCAB, "intensities": INTENSITY, "functions": FUNCTIONS,
            "table": {k: list(v) for k, v in TABLE.items()}, "families": FAMILY}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    return {"version": VOCAB_VERSION, "sha": sha, **body}


# ---- Perform: delivery -> Audio2Face-3D explicit emotion weights (A2F Mark's 10 emotions) ----
# Kept out of vocabulary()'s hash on purpose: retuning how a delivery LOOKS must not invalidate
# directions people already corrected. Tracks record PERFORM_VERSION + perform_sha() instead.
# Weights are the "clear" level, in the 0.3-0.6 range the LLM intent used live (render PZJPBn);
# written from acting sense, not fitted to anything. Audio2Emotion is never involved.
PERFORM_VERSION = 1
A2F_EMOTIONS = ["amazement", "anger", "cheekiness", "disgust", "fear", "grief", "joy", "outofbreath", "pain",
                "sadness"]
PERFORM = {
    "neutral": {}, "deadpan": {},
    "warm": {"joy": 0.35}, "playful": {"joy": 0.3, "cheekiness": 0.3}, "wry": {"cheekiness": 0.4},
    "teasing": {"cheekiness": 0.5, "joy": 0.15}, "excited": {"joy": 0.45, "amazement": 0.3},
    "amazed": {"amazement": 0.55}, "delighted": {"joy": 0.55}, "earnest": {"sadness": 0.12},
    "confident": {"joy": 0.2, "cheekiness": 0.2}, "proud": {"joy": 0.4, "cheekiness": 0.25},
    "skeptical": {"disgust": 0.25, "cheekiness": 0.15}, "concerned": {"fear": 0.25, "sadness": 0.2},
    "sympathetic": {"sadness": 0.3, "joy": 0.15},
}
PERFORM_SCALE = {"slight": 0.6, "clear": 1.0, "strong": 1.4}
PERFORM_MAX = 0.9


def perform_weights(delivery, intensity):
    k = PERFORM_SCALE.get(intensity or "clear", 1.0)
    return {e: round(min(PERFORM_MAX, w * k), 2) for e, w in PERFORM.get(delivery, {}).items()}


def perform_sha():
    body = {"perform": PERFORM, "scale": PERFORM_SCALE, "max": PERFORM_MAX}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


assert set(PERFORM) == set(VOCAB) and all(e in A2F_EMOTIONS for w in PERFORM.values() for e in w)
