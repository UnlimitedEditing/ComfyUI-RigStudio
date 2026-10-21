"""Rig Studio puppet builder, last step: four rig-puppet-parts sheets -> the finished puppet.

  rig-puppet-pack: RigStudioPuppetPack --card--> SaveImage      (a preview card)
                   RigStudioPuppetPack itself writes the puppet as an M3DS PNG (rigstudio.puppet v2)

The four sheets come from running rig-puppet-parts once per part family (mouth, idle, eyes, brows) on the same
head with the same seed. They arrive through Graydient's image1..image4 fields, positionally in that order
(the Flotsam app sends them that way). Each field can be a real URL, a staged filename in input/, or a mangled
Telegram reference (catalog KI-007 s7-8, s35), all handled by rigstudio/media.resolve.
The packing itself is Rig Studio's puppet_rig.py, unchanged (rigstudio/puppet_job.py).
"""
import asyncio
import os

import numpy as np

from .rigstudio import media as M
from .rigstudio import puppet_job as J


def _card(image):
    import torch

    image = image.convert("RGB")
    image.thumbnail((1280, 1280))  # a preview card, not the full 3x-wide contact sheet
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None,]


class RigStudioPuppetPack:
    CATEGORY = "RigStudio"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("card", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        tip = "part sheet (URL, staged filename or Telegram reference)"
        return {
            "required": {
                "sheet_mouth": ("STRING", {"default": "", "tooltip": "1st: the mouth " + tip}),
                "sheet_idle": ("STRING", {"default": "", "tooltip": "2nd: the idle-expressions " + tip}),
                "sheet_eyes": ("STRING", {"default": "", "tooltip": "3rd: the eyes " + tip}),
                "sheet_brows": ("STRING", {"default": "", "tooltip": "4th: the brows " + tip}),
                "name": ("STRING", {"default": "puppet", "tooltip": "the puppet's name (letters, digits, - and _)"}),
            }
        }

    async def run(self, sheet_mouth, sheet_idle, sheet_eyes, sheet_brows, name):
        import folder_paths

        log = lambda m: print(f"[RigStudioPuppetPack] {m}", flush=True)
        values = {"mouth": sheet_mouth, "idle": sheet_idle, "eyes": sheet_eyes, "brows": sheet_brows}
        paths = {}
        for part, value in values.items():
            if not (value or "").strip():
                continue
            paths[part], _ = await M.resolve(value, f"{part} sheet", log)
        try:
            result = await asyncio.to_thread(J.pack, paths, name, log)
        except J.PackError as e:
            raise ValueError(str(e)) from None

        png = result["png"]
        folder, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            f"rigstudio/{J.clean_name(name)}", folder_paths.get_output_directory(), 1, 1
        )
        file = f"{base}_{counter:05}_.png"
        with open(os.path.join(folder, file), "wb") as f:
            f.write(png)
        log(f"{result['summary']} -> {file}")
        return {
            "ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}]},
            "result": (_card(result["preview"]), result["summary"]),
        }


NODE_CLASS_MAPPINGS = {"RigStudioPuppetPack": RigStudioPuppetPack}
NODE_DISPLAY_NAME_MAPPINGS = {"RigStudioPuppetPack": "Rig Studio: Pack Puppet"}
