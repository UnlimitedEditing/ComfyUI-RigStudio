"""Rig v2 packer: four rig-puppet-parts sheets -> a part-based cut-out puppet (M3DS PNG).

    python analysis/puppet_rig.py samples/puppet/<name>          # expects parts-{mouth,idle,eyes,brows}.png

Every sheet is base + 8 edits of ONE part family (D:\\tripostl\\gen_rigstudio_puppet_parts.py, same POSES order),
so each family's changed pixels mark that part's own region:
  mouth  <- union(mouth, idle) changes: one region; frames = X (base) + A-H speech + 8 idle mouths
  eyes   <- eyes changes: one region for both eyes (lids move together); frames = open (base) + 8 lid poses
  brows  <- brows changes, split into brow_L / brow_R at the face centre (viewer's left/right), so each brow
            animates on its own track; the "skeptical" frame gives brow_L raised / brow_R lowered
Regions are made exclusive (a pixel wanted by two grown regions goes to the nearer part), so no layer ever paints
over a neighbour. Pupils are cut from the base: eye whites = the big near-white blobs in the eye region, pupils =
the holes inside them. For every eye frame whose whites still enclose the pupils ("open-ish" frames) the pupils
are erased to white and a clip mask is stored, so the player can move the base pupils continuously inside the
white; frames where detection fails keep their own drawn pupils ("pupils": "baked").
Writes <dir>/rig/ (layers, puppet.json, preview.png) and <dir>/<name>.rig.png (M3DS).
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3ds  # noqa: E402
from puppet_pack import THRESH, cells, changed_mask  # noqa: E402

FORMAT, VERSION, FILENAME = "rigstudio.puppet", 2, "puppet.rigstudio.json"
POSES = {   # = gen_rigstudio_puppet_parts.py POSES
    "mouth": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "idle": ["neutral", "warm", "grin", "smirk", "confident", "worried", "skeptical", "amazed"],
    "eyes": ["lid25", "lid50", "lid75", "closed", "wide", "squint", "narrow", "shut"],
    "brows": ["up1", "up2", "up3", "frown1", "frown2", "worried", "sad", "skeptical"],
}
GROW = 0.012
FEATHER = 1.2
WHITE = 225          # eye white: EVERY channel above this (a summed threshold also caught the peach skin)


def load_sheets(d):
    sheets = {p: cells(Image.open(os.path.join(d, f"parts-{p}.png"))) for p in POSES}
    base = sheets["mouth"][0]
    for p, cs in sheets.items():
        drift = np.abs(cs[0] - base).sum(-1).mean()
        if drift > 3:
            raise ValueError(f"parts-{p}.png has a different base (mean diff {drift:.1f}): not the same head/size")
    return base, sheets


def keep_largest(mask, n):
    lab, k = ndimage.label(mask)
    if k <= n:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, k + 1))
    return np.isin(lab, 1 + np.argsort(sizes)[::-1][:n])


def part_unions(base, sheets):
    W = base.shape[1]
    grow = max(1, int(GROW * W))
    raw = {
        "mouth": np.logical_or.reduce([changed_mask(f, base) for p in ("mouth", "idle") for f in sheets[p][1:]]),
        "eyes": np.logical_or.reduce([changed_mask(f, base) for f in sheets["eyes"][1:]]),
        "brows": np.logical_or.reduce([changed_mask(f, base) for f in sheets["brows"][1:]]),
    }
    # An edit sometimes spills into a neighbour (live, bluebob2: "narrow"/"shut" eyes also tilted the brows, the
    # strong brow frown also turned the mouth down). Bound each part by the face's own layout, taken from the
    # base: the eye whites (big near-white blobs not touching the border) give the eye band; brows live above
    # it, the mouth below it.
    white = base.min(-1) > WHITE
    lab, n = ndimage.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sizes = ndimage.sum(white, lab, range(1, n + 1))
    eyes_lab = [i for i in (1 + np.argsort(sizes)[::-1]) if i not in border][:2]
    ey = np.nonzero(np.isin(lab, eyes_lab))[0]
    eye_top, eye_bot = int(ey.min()), int(ey.max())
    eh = eye_bot - eye_top
    y = np.arange(base.shape[0])[:, None]
    brow_line = eye_top - int(0.12 * eh)          # brows above, lids below (wide eyes grow up a little)
    mouth_line = eye_bot + int(0.35 * eh)
    raw["brows"] &= y < brow_line
    raw["eyes"] &= (y >= brow_line) & (y < mouth_line)
    raw["mouth"] &= y >= mouth_line
    # ...and between the outer edges of the eye whites: open mouths / the grin also redraw the jaw and chin outline
    # (bluebob2: blobs at x 94-714), which grew the mouth region across the whole head.
    ex = np.nonzero(np.isin(lab, eyes_lab))[1]
    x = np.arange(base.shape[1])[None, :]
    raw["mouth"] &= (x >= ex.min()) & (x <= ex.max())
    # Cheek blush: every edit redraws it slightly differently (bluebob2: blobs at the eye columns in 14 of 16
    # mouth frames), which bridged the mouth region out to both cheeks. It belongs to no part: find it in the
    # base by colour (much redder than skin: R-G > 85) and keep it out of every region.
    rgb = base.astype(np.int32)
    blush = (rgb[..., 0] > 190) & (rgb[..., 0] - rgb[..., 1] > 85)
    blush = keep_largest(ndimage.binary_opening(blush, iterations=2), 2)
    blush = ndimage.binary_dilation(blush, iterations=max(2, int(0.01 * W)))
    for p in raw:
        raw[p] &= ~blush
    # stray changes elsewhere (hair shimmer, a cheek): keep the main blobs only, judged after growing
    want = {"mouth": 1, "eyes": 2, "brows": 2}
    grown = {}
    for p, m in raw.items():
        g = ndimage.binary_fill_holes(ndimage.binary_dilation(m, iterations=grow))
        g = keep_largest(g, want[p])
        raw[p] &= g
        grown[p] = g
    # exclusive: contested pixels go to the part whose own changes are nearest
    names = list(grown)
    dist = np.stack([np.where(grown[p], ndimage.distance_transform_edt(~raw[p]), np.inf) for p in names])
    owner = np.argmin(dist, axis=0)
    anyone = np.isfinite(dist.min(axis=0))
    return {p: anyone & (owner == i) for i, p in enumerate(names)}, raw


def soft(mask):
    return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), FEATHER) * 1.25, 0, 1) * \
        ndimage.binary_dilation(mask, iterations=2)


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def crop_layer(rgb, alpha, box):
    x0, y0, x1, y1 = box
    return Image.fromarray(np.dstack([rgb[y0:y1, x0:x1].astype(np.uint8),
                                      (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)]), "RGBA")


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def scleras(img, region):
    """Per eye (left, right by x): the largest near-white blob inside the eye region."""
    white = (img.min(-1) > WHITE) & region
    lab, k = ndimage.label(white)
    if k < 2:
        return None
    sizes = ndimage.sum(white, lab, range(1, k + 1))
    two = 1 + np.argsort(sizes)[::-1][:2]
    blobs = [lab == i for i in two]
    if min(b.sum() for b in blobs) < 0.0005 * white.size:
        return None
    return sorted(blobs, key=lambda b: np.nonzero(b)[1].mean())


def pupils_of(img, region):
    """-> [(sclera, hole)] per eye, or None when the whites do not fully enclose the pupils."""
    sc = scleras(img, region)
    if sc is None:
        return None
    out = []
    for s in sc:
        filled = ndimage.binary_fill_holes(s)
        hole = keep_largest(filled & ~s, 1)
        if hole.sum() < 0.00015 * s.size:
            return None
        out.append((s, hole, filled))
    return out


def build(d):
    base, sheets = load_sheets(d)
    H, W = base.shape[:2]
    regions, raw = part_unions(base, sheets)
    out = os.path.join(d, "rig")
    os.makedirs(out, exist_ok=True)
    files = {"base.png": png(Image.fromarray(base.astype(np.uint8)))}
    parts = {}

    def add_part(name, region, frames):
        alpha = soft(region)
        box = bbox(alpha > 0.01)
        entry = {"box": box, "frames": {}}
        for key, img in frames:
            f = f"{name}_{key}.png"
            files[f] = png(crop_layer(img, alpha, box))
            entry["frames"][key] = f
        parts[name] = entry
        return entry, alpha, box

    # mouth: rest X, speech A-H, idle set
    mouth = [("X", base)] + list(zip(POSES["mouth"], sheets["mouth"][1:])) + \
            [(f"idle_{k}", f) for k, f in zip(POSES["idle"], sheets["idle"][1:])]
    e, _, _ = add_part("mouth", regions["mouth"], mouth)
    e.update(rest="X", speech=POSES["mouth"], idle=[f"idle_{k}" for k in POSES["idle"]])

    # brows, split at the centre between the two brow blobs
    lab, _ = ndimage.label(regions["brows"])
    xs = np.nonzero(regions["brows"])[1]
    cx = int((np.percentile(xs, 25) + np.percentile(xs, 75)) / 2)
    for side, sel in (("brow_L", np.arange(W) < cx), ("brow_R", np.arange(W) >= cx)):
        e, _, _ = add_part(side, regions["brows"] & sel[None, :],
                           [("rest", base)] + list(zip(POSES["brows"], sheets["brows"][1:])))
        e["rest"] = "rest"

    # eyes + pupils
    eye_frames = [("open", base)] + list(zip(POSES["eyes"], sheets["eyes"][1:]))
    base_pupils = pupils_of(base, regions["eyes"])
    pupil_info, clips = [], {}
    processed = []
    for key, img in eye_frames:
        img = img.copy()
        found = pupils_of(img, regions["eyes"]) if base_pupils else None
        if found:
            clip = np.zeros(img.shape[:2], bool)
            for s, hole, filled in found:
                white_col = np.median(img[s], axis=0)
                grow_hole = ndimage.binary_dilation(hole, iterations=1) & filled
                img[grow_hole] = white_col
                clip |= filled
            clips[key] = clip
        processed.append((key, img))
    e, alpha, box = add_part("eyes", regions["eyes"], processed)
    e["rest"] = "open"
    e["pupils"] = {}
    for key, clip in clips.items():
        f = f"eyes_{key}_clip.png"
        c = (clip & (alpha > 0.5))[box[1]:box[3], box[0]:box[2]]
        a = (c * 255).astype(np.uint8)     # white + alpha: the player uses it directly as a canvas clip
        files[f] = png(Image.fromarray(np.dstack([np.full_like(a, 255)] * 3 + [a]), "RGBA"))
        e["pupils"][key] = f
    for k, _ in eye_frames:
        e["pupils"].setdefault(k, "baked")
    if base_pupils:
        for side, (s, hole, filled) in zip(("L", "R"), base_pupils):
            pb = bbox(hole)
            sprite = crop_layer(base, hole.astype(np.float32), pb)
            f = f"pupil_{side}.png"
            files[f] = png(sprite)
            sb = bbox(filled)
            pupil_info.append({"side": side, "file": f, "box": pb,
                               # how far the pupil can travel inside the white, px, from rest
                               "range": [int(max(2, ((sb[2] - sb[0]) - (pb[2] - pb[0])) / 2 * 0.85)),
                                         int(max(1, ((sb[3] - sb[1]) - (pb[3] - pb[1])) / 2 * 0.7))]})

    puppet = {"format": FORMAT, "version": VERSION, "name": os.path.basename(os.path.abspath(d)), "size": [W, H],
              "style": "flat-cartoon", "base": "base.png", "parts": parts, "pupils": pupil_info,
              "draw_order": ["eyes", "pupils", "brow_L", "brow_R", "mouth"], "mouth_set": "rhubarb",
              "source": {"generator": "rig-puppet-parts v1 (Qwen Image 2.1 edit)", "poses": POSES},
              "expressions": []}
    for f, b in files.items():
        with open(os.path.join(out, f), "wb") as fh:
            fh.write(b)
    with open(os.path.join(out, "puppet.json"), "w") as fh:
        json.dump(puppet, fh, indent=1)
    bundle = {**puppet, "files": {k: base64.b64encode(v).decode() for k, v in files.items()}}
    path = os.path.join(d, f"{puppet['name']}.rig.png")
    m3ds.encode(json.dumps(bundle).encode(), FILENAME).save(path, format="PNG", compress_level=6)
    preview(out, puppet, base)
    print(f"{path}: {os.path.getsize(path) // 1024} KB")
    for n, p in parts.items():
        extra = f"  pupils procedural in {[k for k, v in p.get('pupils', {}).items() if v != 'baked']}" if n == "eyes" else ""
        print(f"  {n:<7} box {p['box']}  {len(p['frames'])} frames{extra}")
    print(f"  pupils: {pupil_info}")
    return puppet


def preview(out, puppet, base):
    base_img = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
    tiles = []
    for n, p in puppet["parts"].items():
        for key, f in p["frames"].items():
            im = base_img.copy()
            im.alpha_composite(Image.open(os.path.join(out, f)), tuple(p["box"][:2]))
            d = ImageDraw.Draw(im)
            d.rectangle(p["box"], outline=(255, 0, 0, 255), width=2)
            d.text((8, 8), f"{n}:{key}", fill=(0, 0, 0, 255))
            tiles.append(im.resize((im.width // 3, im.height // 3)))
    cols = 10
    tw, th = tiles[0].size
    grid = Image.new("RGBA", (tw * cols, th * ((len(tiles) + cols - 1) // cols)), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * tw, (i // cols) * th))
    grid.convert("RGB").save(os.path.join(out, "preview.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    build(ap.parse_args().dir)
