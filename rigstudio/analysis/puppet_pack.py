"""Pack a rig-puppet face-set sheet into a cut-out talking-head puppet (M3DS PNG).

    python analysis/puppet_pack.py samples/puppet/faceset-<render>.png --name girl [--render <hash>]

The sheet (D:\\tripostl\\gen_rigstudio_puppet.py) is a 3x3 grid: the base head + 8 full Qwen 2.1 edits, each
changing the MOUTH to one Rhubarb shape and the EYES/BROWS to one state (FRAMES below, same order). Edits are
pixel-aligned with the base (probe v9orPD), so:
  1. diff every frame against the base; ignore faint outline shimmer (THRESH) and specks
  2. the changed pixels form two bands -- upper face (eyes + brows) and mouth -- split at the widest empty
     row gap between them
  3. ONE soft mask per region (union over all frames, dilated + feathered), so every frame of a region
     covers exactly the same area -> any mouth frame can replace any other over the base
  4. layers = each frame's pixels inside its region mask, cropped to the region box; the base's own
     pixels there are the rest frames (mouth X, upper "open")
Writes samples/puppet/<name>/ (layer PNGs + puppet.json + preview.png) and <name>.puppet.png (M3DS).
This is the prototype of the packing node that will end the rig-puppet workflow on Graydient.
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ds  # noqa: E402

FORMAT, VERSION, FILENAME = "rigstudio.puppet", 1, "puppet.rigstudio.json"
# same order as gen_rigstudio_puppet.py FRAMES: (mouth shape, upper-face state)
FRAMES = [("A", "blink"), ("B", "half"), ("C", "brows_up"), ("D", "look_left"),
          ("E", "frown"), ("F", "look_right"), ("G", "squint"), ("H", "worried")]
THRESH = 90       # summed |RGB diff|: real edits change flat colours by hundreds; outline shimmer stays below
MIN_BLOB = 0.0004  # fraction of the frame: smaller changed blobs are shimmer specks
GROW = 0.012      # region mask dilation, fraction of the frame width
FEATHER = 1.5     # px of gaussian softening on the mask edge


def cells(sheet):
    W, H = sheet.size[0] // 3, sheet.size[1] // 3
    return [np.asarray(sheet.crop((c * W, r * H, (c + 1) * W, (r + 1) * H)).convert("RGB"), dtype=np.int16)
            for r in range(3) for c in range(3)]


def changed_mask(frame, base):
    m = np.abs(frame - base).sum(-1) > THRESH
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes >= MIN_BLOB * m.size)[0])
        m = m & keep
    return m


def split_rows(union):
    """Row index between the upper-face band and the mouth band: the middle of the widest empty run of rows
    between the first and last changed rows."""
    rows = np.nonzero(union.any(1))[0]
    if len(rows) < 2:
        raise ValueError("no changes found between the frames and the base")
    gaps = np.diff(rows)
    k = int(np.argmax(gaps))
    if gaps[k] < 3:
        raise ValueError("the mouth and eye changes touch: cannot split the regions")
    return int((rows[k] + rows[k + 1]) // 2)


def region_mask(union, W):
    grown = ndimage.binary_dilation(union, iterations=max(1, int(GROW * W)))
    grown = ndimage.binary_fill_holes(grown)
    soft = ndimage.gaussian_filter(grown.astype(np.float32), FEATHER)
    return np.clip(soft * 1.2, 0, 1)


def to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def pack(sheet_path, name, render=""):
    sheet = Image.open(sheet_path)
    cs = cells(sheet)
    base, frames = cs[0], cs[1:]
    H, W = base.shape[:2]
    masks = [changed_mask(f, base) for f in frames]
    union = np.logical_or.reduce(masks)
    y_split = split_rows(union)
    upper_u, mouth_u = union.copy(), union.copy()
    upper_u[y_split:] = False
    mouth_u[:y_split] = False
    out_dir = os.path.join(os.path.dirname(sheet_path), name)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(base.astype(np.uint8)).save(os.path.join(out_dir, "base.png"))

    parts, blobs = {}, {"base.png": to_png_bytes(Image.fromarray(base.astype(np.uint8)))}
    for part, u, rest, keys in (("upper", upper_u, "open", [f[1] for f in FRAMES]),
                                ("mouth", mouth_u, "X", [f[0] for f in FRAMES])):
        alpha = region_mask(u, W)
        # clamp at the split so the two regions never overlap: layers then swap independently
        if part == "upper":
            alpha[y_split:] = 0
        else:
            alpha[:y_split] = 0
        ys, xs = np.nonzero(alpha > 0.01)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        a = (alpha[box[1]:box[3], box[0]:box[2]] * 255).astype(np.uint8)
        entries = {}
        for key, src in [(rest, base)] + list(zip(keys, frames)):
            rgb = src[box[1]:box[3], box[0]:box[2]].astype(np.uint8)
            layer = Image.fromarray(np.dstack([rgb, a]), "RGBA")
            fname = f"{part}_{key}.png"
            layer.save(os.path.join(out_dir, fname))
            blobs[fname] = to_png_bytes(layer)
            entries[key] = fname
        own = {k: round(float((m & u).sum()) / max(1, u.sum()), 3) for k, m in zip(keys, masks)}
        parts[part] = {"box": box, "rest": rest, "frames": entries,
                       "coverage": {k: float(v) for k, v in own.items()}}   # share of the region each frame actually changed (0 = edit missed)
    puppet = {"format": FORMAT, "version": VERSION, "name": name, "size": [W, H], "style": "flat-cartoon",
              "source": {"sheet": os.path.basename(sheet_path), "render": render,
                         "generator": "rig-puppet v1 (Qwen Image 2.1 edit)"},
              "base": "base.png", "split_y": y_split, "parts": parts,
              "mouth_set": "rhubarb", "expressions": []}
    with open(os.path.join(out_dir, "puppet.json"), "w") as f:
        json.dump(puppet, f, indent=1)
    bundle = {**puppet, "files": {k: base64.b64encode(v).decode() for k, v in blobs.items()}}
    m3ds_path = os.path.join(os.path.dirname(sheet_path), f"{name}.puppet.png")
    m3ds.encode(json.dumps(bundle).encode(), FILENAME).save(m3ds_path, format="PNG", compress_level=6)
    preview(out_dir, puppet, base)
    print(f"{m3ds_path}: {os.path.getsize(m3ds_path) // 1024} KB, split at y={y_split}")
    for part, p in parts.items():
        weak = [k for k, c in p["coverage"].items() if c < 0.05]
        print(f"  {part:<5} box {p['box']}  coverage {p['coverage']}" + (f"  WEAK: {weak}" if weak else ""))
    return puppet


def preview(out_dir, puppet, base):
    """Every mouth frame and every upper frame composited over the base, labelled grid."""
    from PIL import ImageDraw
    base_img = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
    tiles = []
    for part, p in puppet["parts"].items():
        for key, fname in p["frames"].items():
            im = base_img.copy()
            layer = Image.open(os.path.join(out_dir, fname))
            im.alpha_composite(layer, tuple(p["box"][:2]))
            d = ImageDraw.Draw(im)
            d.rectangle(p["box"], outline=(255, 0, 0, 255), width=2)
            d.text((10, 10), f"{part}:{key}", fill=(0, 0, 0, 255))
            tiles.append(im.resize((im.width // 2, im.height // 2)))
    cols = 9
    tw, th = tiles[0].size
    grid = Image.new("RGBA", (tw * cols, th * ((len(tiles) + cols - 1) // cols)), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * tw, (i // cols) * th))
    grid.convert("RGB").save(os.path.join(out_dir, "preview.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet")
    ap.add_argument("--name", required=True)
    ap.add_argument("--render", default="")
    a = ap.parse_args()
    pack(a.sheet, a.name, a.render)


if __name__ == "__main__":
    main()
