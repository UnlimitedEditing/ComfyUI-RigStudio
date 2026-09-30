"""Rig Studio direction-sheet nodes: Direct (audio + brief -> sheet) and Perform (sheet -> track).

  rig-direct:  RigStudioDirect --sheet_json--> RigStudioSaveSheet (M3DS PNG)  + card IMAGE -> SaveImage
  rig-perform: RigStudioPerform --track_data--> SaveImage

The direction sheet (rigstudio/analysis/sheet.py) is one lossless PNG carrying the transcript with word
timings, the brief, and per-sentence function / delivery / intensity / locked, plus seed, vocabulary and
A2F model/engine ids. Users correct it in Sheet Studio; re-directing an edited sheet keeps every locked
line; Perform is deterministic from the sheet (fixed delivery -> A2F emotion table + the sheet's seed).

Inputs are sorted by content, not by field: a PNG is a sheet, anything else is audio, so the same
Graydient media fields serve both (URL / staged filename / Telegram reference — catalog KI-007 §7-8).
Emotion comes from the director reading the TEXT; Audio2Emotion is never used.
"""
import asyncio
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rigstudio", "analysis"))
import director  # noqa: E402
import director_fn  # noqa: E402
import sheet as S  # noqa: E402
import vocab  # noqa: E402

from .nodes_a2f import build_track, encode_string_as_image  # noqa: E402
from .rigstudio import llm_runtime as L  # noqa: E402
from .rigstudio import media as M  # noqa: E402

LLMS = ["qwen3:14b", "none"]


def _clean_brief(text):
    return " ".join((text or "").split())[:1500]


def sheet_card(sheet, title):
    """Readable summary image of a sheet (chat users can't read the data PNG)."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, bold = ImageFont.load_default(size=17), ImageFont.load_default(size=21)
    except TypeError:                                     # Pillow < 10.1: fixed bitmap font
        font = bold = ImageFont.load_default()
    rows = [(title, bold, (20, 20, 24))]
    if sheet.get("brief"):
        rows.append((f"Brief: {sheet['brief'][:150]}", font, (90, 90, 100)))
    rows.append(("", font, (0, 0, 0)))
    colour = {"warm": (190, 70, 45), "fun": (125, 70, 200), "up": (170, 120, 0), "firm": (40, 95, 190),
              "doubt": (25, 125, 110), "neutral": (120, 125, 135)}
    for s in sheet["sentences"]:
        d = s["delivery"] + (f" {s['intensity']}" if s.get("intensity") else "")
        lock = " [locked]" if s["locked"] else ""
        text = s["text"] if len(s["text"]) <= 78 else s["text"][:75] + "..."
        rows.append((f"{int(s['t0'] // 60)}:{s['t0'] % 60:04.1f}  {d:<17} {text}{lock}", font,
                     colour.get(vocab.FAMILY.get(s["delivery"], "neutral"), (0, 0, 0))))
    w, lh = 1100, 24
    img = Image.new("RGB", (w, 40 + lh * len(rows)), (250, 249, 245))
    dr = ImageDraw.Draw(img)
    for k, (t, f, c) in enumerate(rows):
        dr.text((20, 20 + k * lh), t, font=f, fill=c)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


class RigStudioDirect:
    """Audio (+ optional plain-English brief) -> direction sheet. Give it an existing sheet PNG instead
    to RE-direct: the sheet's transcript is reused (no Whisper) and every locked line is kept."""
    CATEGORY = "RigStudio"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet_json", "card", "summary")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_url": ("STRING", {"default": "", "tooltip": "speech audio or a direction sheet PNG (URL / input file)"}),
            "media_url": ("STRING", {"default": "", "tooltip": "second media input (same rules)"}),
            "media_filename": ("STRING", {"default": "", "tooltip": "third media input (same rules)"}),
            "brief": ("STRING", {"multiline": True, "default": "",
                                 "tooltip": "optional plain-English note about the speaker / piece; empty = the sheet's brief"}),
            "llm": (LLMS, {"default": "qwen3:14b", "tooltip": "'none' = transcript-only sheet (for annotating by hand)"}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 2 ** 31 - 1,
                             "tooltip": "LLM sampling seed; also stored in the sheet for Perform's gestures"}),
            "whisper_model": (["medium", "large-v3", "small"], {"default": "medium"}),
        }}

    async def run(self, audio_url, media_url, media_filename, brief, llm, seed, whisper_model):
        T, steps = time.time(), []
        log = lambda m: (steps.append(m), print(f"[RigStudioDirect] {m}", flush=True))
        got_sheet, got_audio = await M.sort_inputs(
            {"audio_url": audio_url, "media_url": media_url, "media_filename": media_filename}, log)
        brief = _clean_brief(brief)
        if got_sheet:
            with open(got_sheet[0], "rb") as f:
                sheet = S.loads(f.read())
            if brief:
                sheet["brief"] = brief
            sheet["seed"] = int(seed)
            locked = sum(s["locked"] for s in sheet["sentences"])
            log(f"re-directing a sheet: {len(sheet['sentences'])} lines, {locked} locked (kept)"
                + ("; audio input ignored, the sheet's transcript is used" if got_audio else ""))
        elif got_audio:
            path, url = got_audio
            lyrics = await self.transcribe(path, whisper_model, log)
            sheet = S.new_sheet(lyrics, audio_url=url, audio_sha256=M.sha256_file(path), brief=brief, seed=seed)
            if not url:
                log("audio did not arrive as a public URL: the sheet stores its checksum only, "
                    "so Perform needs the audio again")
        else:
            raise ValueError("Rig Studio Direct needs speech audio (or a direction sheet PNG to re-direct)")

        if llm != "none" and sheet["sentences"]:
            directed, meta = await self.direct(sheet, llm, seed, log)
            S.apply_direction(sheet, directed, meta)
        issues = S.validate(sheet)
        summary = {"sentences": len(sheet["sentences"]), "locked": sum(s["locked"] for s in sheet["sentences"]),
                   "deliveries": {}, "issues": issues, "total_s": round(time.time() - T, 1), "steps": steps}
        for s in sheet["sentences"]:
            summary["deliveries"][s["delivery"]] = summary["deliveries"].get(s["delivery"], 0) + 1
        print(json.dumps(summary), flush=True)
        title = f"Rig Studio direction sheet: {len(sheet['sentences'])} lines, directed by {llm}"
        return (S.dumps(sheet), sheet_card(sheet, title), json.dumps(summary))

    async def transcribe(self, path, model_size, log):
        import folder_paths
        from .rigstudio import runtime as R
        from .rigstudio import transcribe_core as TC
        t0 = time.time()
        wav, duration = await R.fetch_audio_16k(path, log)
        staged = os.path.join(folder_paths.models_dir, "whisper", model_size)
        src = staged if os.path.isfile(os.path.join(staged, "model.bin")) else model_size
        model = await asyncio.to_thread(TC.load_model, src)
        data = await asyncio.to_thread(TC.transcribe, model, wav, None)
        data.setdefault("duration", round(duration, 2))
        data.setdefault("transcriber", {"engine": "faster-whisper", "model": model_size})
        del model                                            # free VRAM before the LLM loads
        gc.collect()
        log(f"transcribed {duration:.1f}s -> {len(data['timeline'])} sentences ({model_size}, "
            f"{'staged' if src == staged else 'hub download'}) in {time.time() - t0:.1f}s")
        return data

    async def direct(self, sheet, llm, seed, log):
        exe = await L.ensure_ollama(log)
        models_dir, missing = L.prepare_models(llm, log)
        async with L.OllamaServer(exe, models_dir, log) as url:
            if missing:
                await L.pull(url, llm, log)
            director.OLLAMA = url
            t0 = time.time()
            res = await asyncio.to_thread(director_fn.direct, S.lyrics_of(sheet), llm, 8, None,
                                          sheet.get("brief", ""), S.locked_functions(sheet), seed)
            log(f"directed {len(res['sentences'])} lines with {llm} in {time.time() - t0:.1f}s, "
                f"parsed {res['parsed_lines']}")
        meta = {"model": llm, "director": "director_fn v3", "engine": f"ollama {L.OLLAMA_VERSION}",
                "seed": int(seed), "parsed_lines": res["parsed_lines"], "brief_used": bool(sheet.get("brief")),
                "reply": res["reply"][:20000]}
        return res["sentences"], meta


class RigStudioSaveSheet:
    """Write the sheet as a lossless M3DS PNG (same container as Meshsmuggler / YuE2 score PNGs)."""
    CATEGORY = "RigStudio"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sheet_json": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "rigstudio/direction_sheet"})}}

    def save(self, sheet_json, filename_prefix):
        import folder_paths
        sheet = S.loads(sheet_json.encode("utf-8"))
        image = S.to_png(sheet)
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), image.width, image.height)
        file = f"{name}_{counter:05}_.png"
        image.save(os.path.join(folder, file), format="PNG", compress_level=6)
        print(f"[RigStudioSaveSheet] {len(sheet_json)} bytes -> {file} ({image.width}x{image.height})", flush=True)
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class RigStudioPerform:
    """Direction sheet (+ the audio, if the sheet has no public audio URL) -> expression track.
    Deterministic: delivery -> A2F emotion weights from a fixed table (vocab.PERFORM), gestures seeded
    by the sheet's seed, then the same A2F + gesture path as RigStudioBuildTrack."""
    CATEGORY = "RigStudio"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("track_data", "summary")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sheet_url": ("STRING", {"default": "", "tooltip": "direction sheet PNG, or audio (sorted by content)"}),
            "media_url": ("STRING", {"default": "", "tooltip": "second media input (same rules)"}),
            "media_filename": ("STRING", {"default": "", "tooltip": "third media input (same rules)"}),
            "engine_url": ("STRING", {"default": "", "tooltip": "empty = published Ampere+ engine; 'build' = build in-job"}),
            "seed_override": ("INT", {"default": -1, "min": -1, "max": 2 ** 31 - 1, "tooltip": "-1 = the sheet's seed"}),
        }}

    async def run(self, sheet_url, media_url, media_filename, engine_url, seed_override):
        log = lambda m: print(f"[RigStudioPerform] {m}", flush=True)
        got_sheet, got_audio = await M.sort_inputs(
            {"sheet_url": sheet_url, "media_url": media_url, "media_filename": media_filename}, log)
        if not got_sheet:
            raise ValueError("Rig Studio Perform needs a direction sheet PNG (send it as a file, not a photo)")
        with open(got_sheet[0], "rb") as f:
            sheet = S.loads(f.read())
        for issue in S.validate(sheet):
            log(f"sheet issue: {issue}")
        want = (sheet.get("audio") or {}).get("sha256") or ""
        if got_audio:
            audio_path, audio_name = got_audio[0], os.path.basename((got_audio[1] or got_audio[0]).split("?")[0])
        elif (sheet.get("audio") or {}).get("url"):
            audio_path, _ = await M.resolve(sheet["audio"]["url"], "sheet audio url", log)
            audio_name = os.path.basename(sheet["audio"]["url"].split("?")[0])
        else:
            raise ValueError("This sheet has no public audio URL: send the speech audio together with the sheet")
        have = M.sha256_file(audio_path)
        match = (have == want) if want else None
        if match is False:
            log("WARNING: the audio's checksum differs from the sheet's: timings may not line up")
        seed = seed_override if seed_override >= 0 else int(sheet.get("seed", 7))
        intent = S.to_intent(sheet)
        text, summary = await build_track(audio_path, intent, engine_url, seed, audio_name=audio_name,
                                          log_name="RigStudioPerform")
        track = json.loads(text)
        track["direction"] = {"sheet_version": sheet.get("version"), "vocabulary": sheet["vocabulary"].get("sha"),
                              "perform_version": vocab.PERFORM_VERSION, "perform_sha": vocab.perform_sha(),
                              "seed": seed, "audio_sha256": have, "audio_matches_sheet": match,
                              "lines": [{"t0": s["t0"], "t1": s["t1"], "delivery": s["delivery"],
                                         "intensity": s.get("intensity")} for s in sheet["sentences"]]}
        summary.update({"seed": seed, "audio_matches_sheet": match, "lines": len(sheet["sentences"])})
        print(json.dumps(summary), flush=True)
        return (encode_string_as_image(json.dumps(track, separators=(",", ":"))), json.dumps(summary))


NODE_CLASS_MAPPINGS = {"RigStudioDirect": RigStudioDirect, "RigStudioSaveSheet": RigStudioSaveSheet,
                       "RigStudioPerform": RigStudioPerform}
NODE_DISPLAY_NAME_MAPPINGS = {"RigStudioDirect": "Rig Studio: direct (audio + brief -> direction sheet)",
                              "RigStudioSaveSheet": "Rig Studio: save direction sheet PNG",
                              "RigStudioPerform": "Rig Studio: perform (direction sheet -> expression track)"}
