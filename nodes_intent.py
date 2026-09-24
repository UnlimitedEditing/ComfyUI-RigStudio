"""Rig Studio intent nodes (Job A): transcript -> LLM prompt -> compact intent string.

  TranscribeAudioFromURL (ComfyUI-TripoSG) --lyrics_json--> RigStudioIntentPrompt --prompt-->
  HFTextGenerate (ComfyUI-TripoSG, Qwen2.5-7B) --response--> RigStudioIntentPack --> intent

The intent string is what RigStudioBuildTrack (Job B) consumes:
    S <t0> <t1> [q|wq] <emotion> <weight> ... | none   (letters, digits, dots, spaces only)
Emotion weights are authored by an LLM from the *text* — Audio2Emotion is never involved.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rigstudio", "analysis"))
from intent import EMOTIONS, PROMPT, _UNSAFE, parse_reply  # noqa: E402

from .nodes_a2f import encode_string_as_image  # noqa: E402

WH = re.compile(r"^(?:\W*(?:well|so|and|but|okay|ok|now|oh)\W+)*(what|why|how|where|who|whom|which|when)\b", re.I)


def _segments(lyrics_json):
    data = json.loads(lyrics_json or "{}")
    segs = [s for s in data.get("timeline", []) if s.get("type", "lyric") == "lyric" and s.get("text", "").strip()]
    return [{"t0": float(s["start"]), "t1": float(s["end"]), "text": s["text"].strip()} for s in segs]


class RigStudioIntentPrompt:
    CATEGORY = "RigStudio"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lyrics_json": ("STRING", {"forceInput": True})}}

    def run(self, lyrics_json):
        segs = _segments(lyrics_json)
        lines = "\n".join(f"S{i} at {s['t0']:.1f}s - {_UNSAFE.sub(' ', s['text'])}" for i, s in enumerate(segs))
        return (PROMPT.replace("LINES", lines or "S0 at 0.0s - (no speech)"),)


class RigStudioIntentPack:
    CATEGORY = "RigStudio"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("intent_data", "intent")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lyrics_json": ("STRING", {"forceInput": True}),
                             "llm_reply": ("STRING", {"forceInput": True})}}

    def run(self, lyrics_json, llm_reply):
        segs = _segments(lyrics_json)
        try:
            per = parse_reply(llm_reply, len(segs)) if segs else []
            err = None
        except ValueError as e:          # unusable reply -> neutral intent, but still ship timing
            per, err = [{} for _ in segs], str(e)
        parts = []
        for s, e in zip(segs, per):
            q = "wq" if s["text"].endswith("?") and WH.search(s["text"]) else ("q" if s["text"].endswith("?") else "")
            emo = " ".join(f"{k} {round(v, 2)}" for k, v in e.items() if k in EMOTIONS) or "none"
            parts.append(" ".join(x for x in ["S", f"{s['t0']:.2f}", f"{s['t1']:.2f}", q, emo] if x))
        intent = " ".join(parts)
        payload = {"format": "rigstudio.intent", "version": 0, "intent": intent,
                   "segments": [{**s, "e": e} for s, e in zip(segs, per)],
                   "llm_reply": llm_reply[:20000], "parse_error": err}
        print(f"[RigStudioIntentPack] {len(segs)} segments, "
              f"{sum(1 for e in per if e)} emotional{'; parse error: ' + err if err else ''}", flush=True)
        return (encode_string_as_image(json.dumps(payload, ensure_ascii=False)), intent)


NODE_CLASS_MAPPINGS = {"RigStudioIntentPrompt": RigStudioIntentPrompt, "RigStudioIntentPack": RigStudioIntentPack}
NODE_DISPLAY_NAME_MAPPINGS = {"RigStudioIntentPrompt": "Rig Studio: intent prompt from transcript",
                              "RigStudioIntentPack": "Rig Studio: pack LLM reply into intent"}
