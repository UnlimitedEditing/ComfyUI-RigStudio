"""Four rig-puppet-parts sheets -> one finished rigstudio.puppet v2 puppet (an M3DS PNG).

No ComfyUI imports here, so it can be run and tested on its own. The work is Rig Studio's own packer
(analysis/puppet_rig.py, copied byte-identical): every sheet is the base head plus 8 edits of ONE part family,
so the changed pixels mark that part's region; regions are made exclusive, eyes and pupils are separated, and the
layers are packed with the base into one bundle (see that file's header). Output is checked against the puppet
the same sheets make on a PC (identical bundle).

Inputs are positional: mouth, idle, eyes, brows, the same order rig-puppet-parts is run in.
"""
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))

PARTS = ("mouth", "idle", "eyes", "brows")
MAX_SHEET_PIXELS = 80_000_000
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


class PackError(ValueError):
    """A problem the user can act on (wrong or mismatched sheets), not a bug."""


def clean_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "").strip()).strip("-")[:40] or "puppet"


def _check_sheet(part, path):
    from PIL import Image

    with open(path, "rb") as f:
        if f.read(8) != PNG_MAGIC:
            raise PackError(f"The {part} sheet is not a PNG image (it may have been recompressed on the way).")
    with Image.open(path) as im:
        w, h = im.size
    if w * h > MAX_SHEET_PIXELS:
        raise PackError(f"The {part} sheet is too large ({w}x{h}).")
    if w < 90 or h < 90 or w % 3 or h % 3:
        raise PackError(f"The {part} sheet is {w}x{h}: a part sheet is a 3x3 grid, so both sides must divide by 3.")
    return w, h


def pack(sheet_paths, name="puppet", log=print):
    """{part: path to its 3x3 sheet PNG} -> {"png": bytes of the M3DS puppet PNG, "preview": PIL image, "summary": str}."""
    missing = [p for p in PARTS if not sheet_paths.get(p)]
    if missing:
        raise PackError(f"Missing sheet(s): {', '.join(missing)}. Send the mouth, idle, eyes and brows sheets in that order.")
    sizes = {p: _check_sheet(p, sheet_paths[p]) for p in PARTS}
    if len(set(sizes.values())) != 1:
        shown = ", ".join(f"{p} {w}x{h}" for p, (w, h) in sizes.items())
        raise PackError(f"The four sheets are not the same size ({shown}), so they are not from the same head.")

    import puppet_rig  # the packer, run unchanged

    name = clean_name(name)
    root = tempfile.mkdtemp(prefix="rigstudio-pack-")
    try:
        work = os.path.join(root, name)  # the packer names the puppet after its folder
        os.makedirs(work)
        for part in PARTS:
            shutil.copyfile(sheet_paths[part], os.path.join(work, f"parts-{part}.png"))
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                puppet_rig.build(work)
        except ValueError as e:
            if "zero-size array" in str(e) or "no identity" in str(e):
                # a sheet whose edits changed nothing: not the output of rig-puppet-parts
                detail = "No differences were found between the head and its edits, so these do not look like rig-puppet-parts sheets."
            else:
                detail = str(e)
            raise PackError(
                f"{detail} Use the four sheets from the same render run (same head, same seed), in the order mouth, idle, eyes, brows."
            ) from None
        for line in out.getvalue().splitlines():
            log(line)
        with open(os.path.join(work, f"{name}.rig.png"), "rb") as f:
            png = f.read()
        from PIL import Image

        preview = Image.open(os.path.join(work, "rig", "preview.png")).convert("RGB")
        preview.load()
        counts = re.findall(r"^\s*(\w+)\s+box .*?(\d+) frames", out.getvalue(), re.M)
        summary = f"{name}: {len(png) // 1024} KB, " + ", ".join(f"{part} {n} frames" for part, n in counts)
        return {"png": png, "preview": preview, "summary": summary}
    finally:
        shutil.rmtree(root, ignore_errors=True)
