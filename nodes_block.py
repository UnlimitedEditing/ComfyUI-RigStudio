"""Rig Studio Block: shot package PNG + blocking script -> compiled shot package PNG + a readable stage card.

Deterministic, no LLM: the directing happens before the job (a Graydient skill, or a person) and arrives as a
blocking script in the prompt. The package carries its own stage (props, cast, sentences with word timings), so
blocking.py compiles without the set files or direction sheets. An empty script returns the stage card only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rigstudio", "analysis"))

import blocking as B  # noqa: E402
import shot_pack as P  # noqa: E402

from .rigstudio import media as M  # noqa: E402


def _save_png(image, prefix, pnginfo=None):
    import folder_paths
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), image.width, image.height)
    file = f"{name}_{counter:05}_.png"
    image.save(os.path.join(folder, file), format="PNG", compress_level=6, pnginfo=pnginfo)
    return {"filename": file, "subfolder": subfolder, "type": "output"}


def card_image(text):
    """The stage card (plus the compile report) as a legible PNG; the raw text also rides along as a tEXt chunk."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:                                   # Pillow < 10.1
        font = ImageFont.load_default()
    wrapped = []
    for line in text.splitlines() or [""]:
        while len(line) > 96:
            cut = line.rfind(" ", 0, 96)
            cut = cut if cut > 40 else 96
            wrapped.append(line[:cut]); line = "    " + line[cut:].lstrip()
        wrapped.append(line)
    lh = 26
    img = Image.new("RGB", (1180, 40 + lh * len(wrapped)), (250, 248, 242))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(wrapped):
        colour = (176, 40, 30) if line.startswith("REJECTED") else (30, 30, 34)
        d.text((24, 20 + i * lh), line, fill=colour, font=font)
    info = PngInfo()
    info.add_text("rigstudio.stage_card", text)
    return img, info


class RigStudioBlockShot:
    CATEGORY = "RigStudio"
    FUNCTION = "run"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_url": ("STRING", {"default": "", "tooltip": "shot package PNG (send it as a file, not a photo)"}),
            "shot_url_alt": ("STRING", {"default": ""}),
            "shot_filename": ("STRING", {"default": ""}),
            "script": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "blocking script, one beat per line; empty = just return the stage card"}),
            "filename_prefix": ("STRING", {"default": "rigstudio/shot"}),
        }}

    async def run(self, shot_url, shot_url_alt, shot_filename, script, filename_prefix):
        log = lambda m: print(f"[RigStudioBlockShot] {m}", flush=True)
        value, label = next(((v, k) for k, v in (("shot_url", shot_url), ("shot_url_alt", shot_url_alt),
                                                  ("shot_filename", shot_filename)) if (v or "").strip()), (None, None))
        if not value:
            raise ValueError("Rig Studio Block needs a shot package PNG (send it as a file, not a photo)")
        path, _ = await M.resolve(value, label, log)
        with open(path, "rb") as f:
            pkg = P.loads(f.read())
        shot, stage = pkg["shot"], pkg["stage"]
        script = (script or "").replace("\r\n", "\n").strip()
        log(f"script: {len(script.splitlines())} line(s) received")
        report = []
        if script:
            shot, errors = B.compile_script(shot, script, stage)
            report.append(f"COMPILED {len(shot['blocking'])} beat(s), {len(errors)} rejected")
            report += [f"REJECTED: {e}" for e in errors]
            report += ["", "BLOCKING SCRIPT (as compiled):"] + shot["blocking"]
            log(report[0])
            for e in errors:
                log(f"rejected: {e}")
        else:
            report.append("No blocking script given: this is the shot's stage card. Block it with a script in the prompt.")
        out = P.make(shot, stage)
        card, info = card_image(B.stage_card(stage, shot.get("brief", "")) + "\n" + "\n".join(report))
        images = [_save_png(P.to_png(out), filename_prefix),
                  _save_png(card, filename_prefix + "_card", info)]
        print(json.dumps({"beats": len(shot.get("blocking", [])), "report": report[:3]}), flush=True)
        return {"ui": {"images": images}}


NODE_CLASS_MAPPINGS = {"RigStudioBlockShot": RigStudioBlockShot}
NODE_DISPLAY_NAME_MAPPINGS = {"RigStudioBlockShot": "Rig Studio: block shot (script -> shot package)"}
