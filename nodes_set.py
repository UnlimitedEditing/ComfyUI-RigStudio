"""Rig Studio set-layer prompts: one room image -> clean plate + per prop group an RGBA cut-out and an object-space
normal map, all pixel-aligned edits (Qwen Image 2.1), in ONE Graydient job (rig-set-layers).

RigStudioSetPrompts turns the user's prop groups ("the fridge, cupboards and stove; the table and chairs") into the
seven edit prompts. Up to MAX_GROUPS groups; missing groups repeat the last one (the packer ignores repeats).
Separator ';' (Graydient's prompt parser mangles | and brackets).
Normal-map wording is Jacob's (2026-09-30): Qwen returns standard object-space normals (R=+X, G=+Y up, B=+Z to viewer),
unlit -- verified on the kitchen table + chairs.
"""
MAX_GROUPS = 3
RGBA = "This is an RGBA format image with transparency. {} The image has an alpha channel and a transparent background."
CLEAN = ("Remove every piece of furniture and every object from this room, and their shadows: {groups}. Keep the "
         "windows, doors and pictures on the walls. Keep everything else exactly as it is: the same walls, floor, "
         "ceiling, corners, skirting boards, perspective, colours, patterns and black outline style. Fill every place "
         "where an object was with the wall or the floor that would be behind it, continuing its pattern.")
CUT = ("Keep only {g}, exactly as drawn, in exactly the same position and size. Remove everything else, including the "
       "walls, the floor, the ceiling, windows, other furniture and all shadows.")
ROOM_NORMALS = ("Paint the scene with object-space normal map colours, bright colours for easy plane orientation "
                "detection including the floor, walls and ceiling planes. The scene is neutral and completely unlit.")
NORMALS = ("Paint {g} with object-space normal map colours, bright colours for easy plane orientation detection. "
           "Keep the exact same shapes, positions and sizes. The objects are neutral and completely unlit, on a plain "
           "black background.")


class RigStudioSetPrompts:
    CATEGORY = "RigStudio"
    # room_normals is APPENDED (v1 of rig-set-layers is wired to the first 7 outputs)
    RETURN_TYPES = ("STRING",) * (2 + 2 * MAX_GROUPS)
    RETURN_NAMES = ("clean",) + tuple(f"{k}_{i}" for i in range(1, MAX_GROUPS + 1) for k in ("cutout", "normals")) +         ("room_normals",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"groups": ("STRING", {"multiline": True, "default": "the furniture",
                                                   "tooltip": "prop groups separated by ';'"})}}

    def run(self, groups):
        gs = [g.strip().strip(".") for g in groups.replace("\n", ";").split(";") if g.strip()] or ["the furniture"]
        gs = (gs + [gs[-1]] * MAX_GROUPS)[:MAX_GROUPS]
        out = [CLEAN.format(groups="; ".join(dict.fromkeys(gs)))]
        for g in gs:
            out += [RGBA.format(CUT.format(g=g)), NORMALS.format(g=g)]
        out.append(ROOM_NORMALS)
        print(f"[RigStudioSetPrompts] groups: {gs}", flush=True)
        return tuple(out)


NODE_CLASS_MAPPINGS = {"RigStudioSetPrompts": RigStudioSetPrompts}
NODE_DISPLAY_NAME_MAPPINGS = {"RigStudioSetPrompts": "Rig Studio: set-layer prompts (clean plate, cut-outs, normals)"}
