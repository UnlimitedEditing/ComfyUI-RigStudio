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


BROW_INK = 200       # summed RGB below this = outline ink
BROW_BLOB = 100      # px: smaller ink blobs are specks, not strokes
BROW_ZONE_GROW = 4   # px of margin around a brow stroke (covers its antialiased edge)
BROW_THICK = 0.12    # radius of a brow stroke, as a fraction of the eye height (lid lines and lashes are thinner)


def thick_cores(ink, radius):
    """-> (label image, count) of the parts of `ink` that are strokes at least 2*radius thick (brows are; lid lines,
    lashes and hair lines are not): the ink opened with a disk of that radius."""
    r = max(2, radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    core = ndimage.binary_opening(ink, structure=(xx * xx + yy * yy) <= r * r)
    return ndimage.label(core, structure=np.ones((3, 3)))


def brow_strokes(frame, geo):
    """Mask of the eyebrow strokes in a full frame.

    A brow stroke is an ink blob that (1) overlaps what the BROW EDITS redraw (`moving`: the changed pixels of every
    brows-sheet frame, grown a little), (2) is SEPARATE from the eye (its own connected blob), (3) sits inside the eye
    columns, and (4) has its bottom edge above the eyes' midline (every real eye element reaches below it).

    (1) is the one that tells a brow from the eye's own drawing. Position cannot: the eye edits (lid25..narrow, shut)
    redraw the brows low, right where another character's upper-lid arc sits (a freckled boy, 2026-10-21: a thin arc
    above each open eye is part of the eye, and a lowered brow in `narrow` covers the same rows). Lid arcs and hair lines
    never change when only the brows are edited; brows always do."""
    top, bot, x_lo, x_hi, moving = geo[:5]
    mid = top + 0.5 * (bot - top)
    y0 = max(0, top - (bot - top))                # brows can sit well above the whites
    y1 = min(frame.shape[0], bot + int(0.1 * (bot - top)))
    ink = frame[y0:y1].astype(np.int32).sum(-1) < BROW_INK
    comp, n = ndimage.label(ink, structure=np.ones((3, 3)))
    out = np.zeros(frame.shape[:2], bool)
    # Thick strokes are brows even where they touch the eye's ink (a lowered brow bridged to the lid line by one
    # anti-aliased pixel makes ONE blob, which the blob rule below cannot see): found by opening, then regrown.
    r = max(2, int(BROW_THICK * (bot - top)))
    cores, cn = thick_cores(ink, r)
    for i, (ys, xs) in enumerate(ndimage.find_objects(cores), start=1):
        bottom, cx = y0 + ys.stop - 1, (xs.start + xs.stop - 1) / 2
        # tested on the core (regrowing bleeds into the lid line); a little slack, a brow tilted down toward the nose
        # reaches past the midline in `narrow`, while a pupil's core ends far below it
        if bottom < mid + 0.1 * (bot - top) and x_lo <= cx <= x_hi:
            out[y0:y1] |= ndimage.binary_dilation(cores == i, iterations=r + 2, mask=ink)
    if n:
        areas = ndimage.sum(ink, comp, range(1, n + 1))
        for i, (ys, xs) in enumerate(ndimage.find_objects(comp), start=1):
            if areas[i - 1] < BROW_BLOB:
                continue
            bottom, cx = y0 + ys.stop - 1, (xs.start + xs.stop - 1) / 2
            if bottom < mid and x_lo <= cx <= x_hi:
                blob = comp == i
                if (moving[y0:y1] & blob).any():
                    out[y0:y1][blob] = True
    return out


def ink_mask(frame):
    return frame.astype(np.int32).sum(-1) < BROW_INK


def brow_zone(base, sheets, geo):
    """The base's own brow footprint, with a small margin: the static brow pixels belong to the brow part.

    Only strokes the brow edits redraw count (see brow_strokes), so hair lines near the brows are not taken.

    Clipped by the eye ink of the base and the eye frames, so the margin never eats the eye's own drawing (lash tips
    of the open eye sit where the eye edits redraw a brow, see eye_frame_brows)."""
    own = brow_strokes(base, geo)
    eye_ink = np.zeros_like(own)
    for f in [base] + list(sheets["eyes"][1:]):
        eye_ink |= ink_mask(f) & ~ndimage.binary_dilation(brow_strokes(f, geo), iterations=1)
    return ndimage.binary_dilation(own, iterations=BROW_ZONE_GROW) & ~ndimage.binary_dilation(eye_ink, iterations=2)


def eye_frame_brows(frame, geo):
    """The eyebrow strokes of ONE eye frame (plus margin), never touching that frame's own eye drawing.

    The eye edits (lid25, lid50, lid75, narrow, shut on bluebob2) redraw the brows inside the eye band. The brows run as
    their own track, so these strokes must not be part of an eye layer: a second brow would show next to the one the
    brow track draws. They are painted over with skin, not made transparent: the base underneath still has the open
    eye's lash tips exactly where a lowered brow is drawn in these frames."""
    strokes = brow_strokes(frame, geo)
    own_eye = ink_mask(frame) & ~ndimage.binary_dilation(strokes, iterations=1)
    return ndimage.binary_dilation(strokes, iterations=BROW_ZONE_GROW) & ~ndimage.binary_dilation(own_eye, iterations=2)


def skin_color(base, region):
    """The flat skin colour inside a region of the base: the commonest colour that is not ink, white or blush."""
    px = base[region].astype(np.int32)
    keep = (px.sum(-1) >= BROW_INK) & (px.min(-1) <= WHITE) & ~((px[:, 0] > 190) & (px[:, 0] - px[:, 1] > 85))
    px = px[keep]
    if len(px) == 0:
        return np.array([240, 190, 165], dtype=base.dtype)
    q = (px // 8).astype(np.int64)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    top = np.bincount(key).argmax()
    return np.median(px[key == top], axis=0).astype(base.dtype)


def eye_region(base, frames, geo, band):
    """The eye layer's region, built from the eye's own STRUCTURE, not from "everything that changed".

    The union of changed pixels dragged in what the model re-rendered softly from one frame to the next (forehead and
    cheek shading, freckles, the nose bridge) and cut the eye whites where a brow zone overlapped them. Instead, per
    eye: the whites (filled, so pupils are inside) of the base and every frame, plus the ink that belongs to the eye
    (touching the whites, or a long lid line / closed-eye curve / lash row near them; short isolated dots are freckles),
    then the convex hull of that with a small margin. Brow strokes never count.
    Returns (region mask, sclera union mask)."""
    top, bot, x_lo, x_hi = geo[:4]
    eh = bot - top
    y_lo, y_hi = band
    H, W = base.shape[:2]
    yy = np.arange(H)[:, None]
    in_band = (yy >= y_lo) & (yy < y_hi)

    def whites(img):
        white = (img.min(-1) > WHITE) & in_band
        lab, n = ndimage.label(white)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        keep = [i for i in range(1, n + 1) if i not in border and (lab == i).sum() > 0.0005 * white.size]
        return np.isin(lab, keep)

    w0 = keep_largest(whites(base), 2)
    near0 = ndimage.binary_dilation(w0, iterations=max(3, int(0.08 * eh)))
    sclera = np.zeros_like(w0)
    ink_all = np.zeros_like(w0)
    for f in [base] + list(frames):
        wf = whites(f)
        lab, n = ndimage.label(wf)
        if n:
            hit = np.unique(lab[near0 & wf])
            sclera |= np.isin(lab, hit[hit > 0])
        brow = ndimage.binary_dilation(brow_strokes(f, geo), iterations=BROW_ZONE_GROW)
        ink_all |= ink_mask(f) & ~brow & in_band
    sclera = ndimage.binary_fill_holes(sclera | w0)
    xs = np.nonzero(sclera)[1]
    width = xs.max() - xs.min()
    grown = ndimage.binary_dilation(sclera, iterations=2)
    reach = ndimage.binary_dilation(sclera, iterations=max(4, int(0.45 * eh)))   # long ink must at least be near
    lab, n = ndimage.label(ink_all, structure=np.ones((3, 3)))
    eye_ink = np.zeros_like(sclera)
    for i, (ys_, xs_) in enumerate(ndimage.find_objects(lab), start=1):
        blob = lab == i
        if blob.sum() < BROW_BLOB:
            continue
        long_ = (xs_.stop - xs_.start) >= 0.2 * width / 2
        if (blob & grown).any() or (long_ and (blob & reach).any()):
            eye_ink |= blob
    struct = sclera | eye_ink
    cx = np.nonzero(w0)[1].mean()
    region = np.zeros_like(struct)
    margin = max(3, int(0.04 * eh))
    from PIL import ImageDraw
    from scipy.spatial import ConvexHull
    xcol = np.arange(W)[None, :]
    for side in (xcol < cx, xcol >= cx):
        pts = np.argwhere(struct & side)
        if len(pts) < 10:
            continue
        hull = pts[ConvexHull(pts).vertices]
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).polygon([(int(x), int(y)) for y, x in hull], fill=255)
        region |= np.asarray(img) > 0
    return ndimage.binary_dilation(region, iterations=margin) & in_band, sclera, ndimage.binary_fill_holes(w0)


def despeckle_skin(img, region):
    """Within a layer region, replace everything that is neither eye white nor ink (and not next to either) by a median
    of itself: broad lid shading survives, the faint hairline creases the model draws differently in every frame
    (they showed as stray arcs that stayed behind when the brows moved) do not."""
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    keep = (crop.min(-1) > WHITE) | (crop.astype(np.int32).sum(-1) < BROW_INK)
    keep = ndimage.binary_dilation(keep, iterations=3)
    smooth = np.stack([ndimage.median_filter(crop[..., c], size=9) for c in range(3)], -1)
    out = img.copy()
    out[y0:y1, x0:x1] = np.where(keep[..., None], crop, smooth)
    return out


def lidded_pupils(img, sclera, open_white, base_pupils, eh):
    """For an eye frame whose lid cuts the whites (so pupils_of fails): -> (pupil mask, clip) or None.

    The pupils of such a frame are drawn by the model at their own size and place, which flickered against the open
    eye (whose pupils the player draws from the base sprite). Find them (ink blobs too thick to be a lid line or lash,
    inside the whites), paint them white, and return the opening (whites + pupils, i.e. below the lid line) as the clip
    so the player can draw the same base pupils here, hidden by the lid."""
    ink = ink_mask(img)
    r = max(3, int(0.12 * eh))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    core = ndimage.binary_opening(ink & ndimage.binary_dilation(sclera, iterations=r), structure=(xx * xx + yy * yy) <= r * r)
    lab, n = ndimage.label(core)
    want = 0.25 * np.mean([hole.sum() for _, hole, _ in base_pupils])
    keep = [i for i in range(1, n + 1) if (lab == i).sum() >= want]
    if len(keep) != 2:
        return None
    pupil = ndimage.binary_dilation(np.isin(lab, keep), iterations=3, mask=ink)
    near = ndimage.binary_dilation(sclera, iterations=4)   # not the white background around the head
    white = (img.min(-1) > WHITE) & near
    opening = ndimage.binary_fill_holes(ndimage.binary_closing(white | pupil, iterations=2)) & near
    lab, n = ndimage.label(opening)
    ids = np.unique(lab[pupil & opening])
    opening = np.isin(lab, ids[ids > 0])   # the opening the pupils sit in, not the crease lines
    if not opening.any():
        return None
    shown, ext = normalised_opening(opening, open_white, ink & ~pupil)
    return pupil, shown, opening, ext


def normalised_opening(opening, open_white, ink=None):
    """The part of the OPEN eye's white that the lid leaves visible, in place of the frame's own white.

    The model redraws the whites a little smaller or shifted in every lidded frame (inner corner 11 px short on the
    freckled boy), which flickers against the open eye. So the whites of a lidded frame are taken from the base eye:
    each column of the base white is visible from the frame's lid line down. Columns the frame's own white does not
    reach continue the lid line with the slope of its nearest columns (a flat carry-over left steps under a slanted
    lid). -> (visible white mask, extension mask: the lid-line strip over those extra columns, to be drawn as ink)."""
    H, W = opening.shape
    out = np.zeros_like(opening)
    ext = np.zeros_like(opening)
    cx = np.nonzero(open_white)[1].mean()
    rows = np.arange(H)[:, None]
    for side in (np.arange(W) < cx, np.arange(W) >= cx):
        cols = np.nonzero(open_white.any(0) & side)[0]
        if not len(cols):
            continue
        top = np.full(W, np.nan)
        for c in cols:
            r = np.nonzero(opening[:, c])[0]
            if len(r):
                top[c] = r.min()
        ok = cols[~np.isnan(top[cols])]
        if not len(ok):
            return opening, ext
        missing = [c for c in cols if np.isnan(top[c])]
        for c in missing:
            near = ok[np.argsort(np.abs(ok - c))[:25]]
            if len(near) >= 8 and np.ptp(near) >= 6:
                k, m = np.polyfit(near, top[near], 1)
                top[c] = k * c + m
            else:
                top[c] = top[ok[np.argmin(np.abs(ok - c))]]
        t = np.where(np.isnan(top), 0, top)
        out |= open_white & side[None, :] & (rows >= np.round(t)[None, :])
        runs = []
        for c in ok:
            y, n = int(top[c]) - 1, 0
            while ink is not None and y >= 0 and ink[y, c] and n < 40:
                y, n = y - 1, n + 1
            runs.append(n)
        thick = max(3, int(np.median(runs))) if runs else 8
        for c in missing:
            y = int(round(t[c]))
            ext[max(0, y - thick):y, c] = True
    return out, ext


def part_unions(base, sheets):
    W = base.shape[1]
    grow = max(1, int(GROW * W))
    raw = {
        "mouth": np.logical_or.reduce([changed_mask(f, base) for p in ("mouth", "idle") for f in sheets[p][1:]]),
        "eyes": np.logical_or.reduce([changed_mask(f, base) for f in sheets["eyes"][1:]]),
        "brows": np.logical_or.reduce([changed_mask(f, base) for f in sheets["brows"][1:]]),
    }
    moving_brows = raw["brows"].copy()   # before the layout bounds below: a lowered brow reaches past the brow line
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
    # The brow part owns the base's own brow footprint (see brow_zone); the strokes the eye edits redraw are
    # painted out of the eye layers instead (see eye_frame_brows in build).
    span = int(ex.max() - ex.min())
    geo = (eye_top, eye_bot, ex.min() - 0.15 * span, ex.max() + 0.15 * span,
           ndimage.binary_dilation(moving_brows, iterations=grow))
    zone = brow_zone(base, sheets, geo)
    raw["eyes"] &= ~zone
    raw["brows"] |= zone
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
    regions = {p: anyone & (owner == i) for i, p in enumerate(names)}
    # the base's brow footprint is the brows' by rule, whatever the distances said
    regions["brows"] |= zone
    regions["mouth"] &= ~zone
    regions["eyes"], sclera, open_white = eye_region(base, sheets["eyes"][1:], geo, (eye_top - int(0.25 * eh), mouth_line))
    regions["eyes"] &= ~(ndimage.binary_dilation(blush, iterations=8) & ~ndimage.binary_dilation(sclera, iterations=2))   # cheek blush belongs to no part
    regions["eyes"] &= ~(zone & ~ndimage.binary_dilation(sclera, iterations=2))   # the brows' footprint stays theirs
    regions["mouth"] &= ~regions["eyes"]
    return regions, raw, geo + (sclera, open_white)


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
    regions, raw, geo = part_unions(base, sheets)
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
    skin = skin_color(base, regions["eyes"])
    ow = geo[6]
    ring = ndimage.binary_dilation(ow, iterations=3) & ~ow
    ringless = (ink_mask(base) & ring).sum() < 0.25 * ring.sum()   # no black outline around the open eye's white
    bp = base.astype(np.int32)
    plain_skin = (bp.sum(-1) >= BROW_INK) & (bp.min(-1) <= WHITE) & ~((bp[..., 0] > 190) & (bp[..., 0] - bp[..., 1] > 85))
    near_skin = np.abs(bp - skin.astype(np.int32)).sum(-1) <= 45
    for key, img in eye_frames:
        img = img.copy()
        if key != "open":
            # the model re-renders skin shading a little differently in every frame: only strong changes are the eye's
            # (but only where the base is plain skin: its thin pale crease lines are not shading, and the frame
            # rightly leaves them out)
            soft_px = (np.abs(img.astype(np.int16) - base.astype(np.int16)).sum(-1) <= THRESH) & near_skin
            img[soft_px] = base[soft_px]
        erase = eye_frame_brows(img, geo)   # the brows run as their own track: no brow stroke in an eye layer
        img[erase] = np.where(plain_skin[erase][:, None], base[erase], skin)
        # a brow that overlapped the eye whites must not leave a bite out of them: the pupils move inside the white
        bite = erase & geo[5] & (img.min(-1) <= WHITE)
        if (img.min(-1) > WHITE)[geo[5]].any() and key in ("open", "lid25", "wide"):
            img[bite] = np.median(img[geo[5] & (img.min(-1) > WHITE)], axis=0)
        img = despeckle_skin(img, regions["eyes"])
        found = pupils_of(img, regions["eyes"]) if base_pupils else None
        if found:
            clip = np.zeros(img.shape[:2], bool)
            for s, hole, filled in found:
                white_col = np.median(img[s], axis=0)
                grow_hole = ndimage.binary_dilation(hole, iterations=1) & filled
                img[grow_hole] = white_col
                clip |= filled
            if key != "open" and ringless:
                # a character whose open eye has no outline: the ink the model draws around the whites of a wide eye
                # (crescents at the sides and top) is a different eye's outline, not part of this one
                around = ndimage.binary_dilation(clip, iterations=2)
                hug = ndimage.binary_dilation(clip, iterations=max(4, int(0.18 * (geo[1] - geo[0]))))   # not freckles, nose
                gone = ndimage.binary_dilation(ink_mask(img) & ~around & hug, iterations=2)   # and its soft edge
                img[gone & ~around & regions["eyes"]] = skin
            clips[key] = clip
        elif base_pupils and key != "open":
            lp = lidded_pupils(img, geo[5], geo[6], base_pupils, geo[1] - geo[0])
            if lp:
                pupil, shown, own, ext = lp
                white_col = np.median(img[(img.min(-1) > WHITE) & geo[5]], axis=0)
                lid_ink = ink_mask(img) & ~pupil
                img[(own & ~shown) & ~lid_ink] = skin       # the frame's own white where the open eye has none
                img[shown & ~lid_ink] = white_col           # the open eye's white: same shape in every frame, no fringe
                ink_col = np.median(img[lid_ink], axis=0) if lid_ink.any() else np.array([20, 15, 15])
                img[ext & ~shown & regions["eyes"]] = ink_col     # the lid line carried on over the columns it did not reach
                clips[key] = shown
                crease = (img.min(-1) > WHITE) & ~ndimage.binary_dilation(shown, iterations=3) & regions["eyes"]
                img[crease] = skin                          # pale crease lines above the lid are not eye white
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
