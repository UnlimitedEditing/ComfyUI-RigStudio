"""Tests for rigstudio/puppet_job.py (the rig-puppet-pack workflow's logic), run without ComfyUI:

    python -m unittest tests.test_puppet_pack -v

The end-to-end test needs the four sheets and the finished puppet from Rig Studio's samples
(RIG_SAMPLES, default D:/rig-studio/samples/puppet/bluebob2) and is skipped without them.
"""
import glob
import json
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rigstudio", "analysis"))

from PIL import Image  # noqa: E402

from rigstudio import puppet_job as J  # noqa: E402

SAMPLES = os.environ.get("RIG_SAMPLES", "D:/rig-studio/samples/puppet/bluebob2")


def solid_sheet(path, size=(300, 300), colour=(200, 120, 60)):
    Image.new("RGB", size, colour).save(path, format="PNG")


class Validation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def sheets(self, **over):
        out = {}
        for p in J.PARTS:
            path = os.path.join(self.dir, f"{p}.png")
            solid_sheet(path)
            out[p] = path
        out.update(over)
        return out

    def test_names_are_made_safe(self):
        self.assertEqual(J.clean_name("../../etc/passwd"), "etc-passwd")
        self.assertEqual(J.clean_name("  My puppet!  "), "My-puppet")
        self.assertEqual(J.clean_name(""), "puppet")
        self.assertEqual(J.clean_name("x" * 100), "x" * 40)

    def test_missing_sheet(self):
        s = self.sheets()
        del s["eyes"]
        with self.assertRaisesRegex(J.PackError, "Missing sheet.*eyes"):
            J.pack(s)

    def test_not_a_png(self):
        bad = os.path.join(self.dir, "mouth.jpg")
        Image.new("RGB", (300, 300)).save(bad, format="JPEG")
        with self.assertRaisesRegex(J.PackError, "not a PNG"):
            J.pack(self.sheets(mouth=bad))

    def test_not_a_three_by_three_grid(self):
        odd = os.path.join(self.dir, "odd-idle.png")
        solid_sheet(odd, size=(301, 300))
        with self.assertRaisesRegex(J.PackError, "3x3"):
            J.pack(self.sheets(idle=odd))

    def test_sheets_of_different_sizes(self):
        other = os.path.join(self.dir, "big-brows.png")
        solid_sheet(other, size=(600, 600))
        with self.assertRaisesRegex(J.PackError, "not the same size"):
            J.pack(self.sheets(brows=other))

    def test_sheets_where_nothing_changed_are_refused_in_plain_words(self):
        # four identical solid sheets: the edits changed nothing, so these cannot be rig-puppet-parts output
        with self.assertRaisesRegex(J.PackError, "No differences were found"):
            J.pack(self.sheets())

    def test_sheets_from_different_heads_are_refused_in_plain_words(self):
        other = os.path.join(self.dir, "other-eyes.png")
        solid_sheet(other, colour=(10, 200, 240))  # same size, a different base
        with self.assertRaisesRegex(J.PackError, "same render run"):
            J.pack(self.sheets(eyes=other))


@unittest.skipUnless(
    all(os.path.isfile(os.path.join(SAMPLES, f)) for f in ("parts-mouth.png", "parts-idle.png", "parts-eyes.png", "parts-brows.png", "bluebob2.rig.png")),
    "Rig Studio sample sheets not found (set RIG_SAMPLES)",
)
class RealSheets(unittest.TestCase):
    def test_reproduces_the_pc_packer_exactly(self):
        import m3ds

        paths = {p: os.path.join(SAMPLES, f"parts-{p}.png") for p in J.PARTS}
        t0 = time.time()
        result = J.pack(paths, "bluebob2", log=lambda m: None)
        took = time.time() - t0
        got = json.loads(m3ds.decode_png_bytes(result["png"])[0])
        want = json.loads(m3ds.decode_png_bytes(open(os.path.join(SAMPLES, "bluebob2.rig.png"), "rb").read())[0])
        self.assertEqual(got, want)  # the same puppet the PC-side packer makes from the same sheets
        self.assertEqual(got["format"], "rigstudio.puppet")
        self.assertEqual(got["version"], 2)
        self.assertEqual(result["preview"].size[0] > 0, True)
        self.assertIn("mouth", result["summary"])
        print(f"\npacked in {took:.1f} s, {len(result['png']) // 1024} KB")


def brow_strokes_in_eye_layers(bundle):
    """{frame: [(top, bottom, left, right, area), ...]} for every eyebrow stroke baked into an eye layer.

    A brow stroke is an ink blob inside the eye columns that is separate from the eye (its own connected blob) and
    whose bottom edge is above the eyes' midline, in full-frame coordinates: every real eye element (outline, lid line,
    lashes, closed-eye curve) reaches below the midline. The face layout comes from the base image alone."""
    import base64
    import io

    import numpy as np
    from scipy import ndimage

    def load(f):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(bundle["files"][f]))).convert("RGBA"))

    base = load(bundle["base"])[..., :3].astype(np.int32)
    white = base.min(-1) > 225
    lab, n = ndimage.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sizes = ndimage.sum(white, lab, range(1, n + 1))
    eyes_lab = [i for i in (1 + np.argsort(sizes)[::-1]) if i not in border][:2]
    ys, xs = np.nonzero(np.isin(lab, eyes_lab))
    eye_top, eye_bot = int(ys.min()), int(ys.max())
    mid = eye_top + 0.5 * (eye_bot - eye_top)
    span = xs.max() - xs.min()
    x_lo, x_hi = xs.min() - 0.15 * span, xs.max() + 0.15 * span

    eyes = bundle["parts"]["eyes"]
    x0, y0, _, _ = eyes["box"]
    found = {}
    for key, f in eyes["frames"].items():
        px = load(f)
        ink = (px[..., :3].astype(np.int32).sum(-1) < 200) & (px[..., 3] > 128)
        comp, k = ndimage.label(ink, structure=np.ones((3, 3)))
        for i in range(1, k + 1):
            yy, xx = np.nonzero(comp == i)
            if len(yy) < 100:
                continue
            top, bottom, left, right = y0 + yy.min(), y0 + yy.max(), x0 + xx.min(), x0 + xx.max()
            if bottom < mid and x_lo <= (left + right) / 2 <= x_hi:
                found.setdefault(key, []).append((int(top), int(bottom), int(left), int(right), len(yy)))
    return found


@unittest.skipUnless(
    all(os.path.isfile(os.path.join(SAMPLES, f)) for f in ("parts-mouth.png", "parts-idle.png", "parts-eyes.png", "parts-brows.png")),
    "Rig Studio sample sheets not found (set RIG_SAMPLES)",
)
class EyeLayersHaveNoBrows(unittest.TestCase):
    """The brows run as their own track, so no eye layer may carry a brow stroke (it would show as a second brow).

    Found 2026-10-21 on bluebob2: the lid25, lid50, lid75, narrow and shut eye frames each carried one, because the
    eye edits redraw the brows inside the eye band and the packer cut that band out as part of the eye layer."""

    @classmethod
    def setUpClass(cls):
        import m3ds

        paths = {p: os.path.join(SAMPLES, f"parts-{p}.png") for p in J.PARTS}
        result = J.pack(paths, "bluebob2", log=lambda m: None)
        cls.bundle = json.loads(m3ds.decode_png_bytes(result["png"])[0])

    def test_no_eye_layer_carries_a_brow_stroke(self):
        self.assertEqual(brow_strokes_in_eye_layers(self.bundle), {})

    def test_the_eye_itself_is_still_there(self):
        # not over-cleaned: every frame keeps its eye structure (lashes, lid line, closed curve) below the midline
        import base64
        import io

        import numpy as np

        for key, f in self.bundle["parts"]["eyes"]["frames"].items():
            px = np.asarray(Image.open(io.BytesIO(base64.b64decode(self.bundle["files"][f]))).convert("RGBA"))
            ink = ((px[..., :3].astype(int).sum(-1) < 200) & (px[..., 3] > 128)).sum()
            self.assertGreater(ink, 800, f"eye frame {key} lost its drawing ({ink} ink pixels)")

NEWCHAR = os.environ.get("RIG_NEWCHAR", "D:/tripostl/test_out/rig-newchar")


@unittest.skipUnless(
    all(glob.glob(os.path.join(NEWCHAR, p, "*.png")) for p in J.PARTS),
    "new-character sheets not found (set RIG_NEWCHAR)",
)
class EyeLayersHaveNoBrowsNewCharacter(unittest.TestCase):
    """The same rule on a second character (freckled boy, 2026-10-21), whose eye drawing differs from bluebob2:
    a thin arc above each open eye is part of the eye, and the lowered brow in `narrow` tilts past the eye midline.
    The first brow rule erased the arcs and missed that brow; this one has to do neither."""

    @classmethod
    def setUpClass(cls):
        import m3ds

        paths = {p: glob.glob(os.path.join(NEWCHAR, p, "*.png"))[0] for p in J.PARTS}
        result = J.pack(paths, "newchar", log=lambda m: None)
        cls.bundle = json.loads(m3ds.decode_png_bytes(result["png"])[0])

    def test_no_brow_stroke_is_left_in_an_eye_layer(self):
        # `wide`'s outline arc is cut by the layer's top edge into two small blobs, which are not brows
        found = brow_strokes_in_eye_layers(self.bundle)
        big = {k: [c for c in v if c[4] >= 400] for k, v in found.items()}
        self.assertEqual({k: v for k, v in big.items() if v}, {})

    def test_the_lowered_brow_in_narrow_is_gone_and_its_lid_line_stays(self):
        import base64
        import io

        import numpy as np
        from scipy import ndimage

        f = self.bundle["parts"]["eyes"]["frames"]["narrow"]
        px = np.asarray(Image.open(io.BytesIO(base64.b64decode(self.bundle["files"][f]))).convert("RGBA"))
        ink = (px[..., :3].astype(int).sum(-1) < 200) & (px[..., 3] > 128)
        lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
        thick = [i for i in range(1, n + 1) if ndimage.distance_transform_edt(lab == i).max() >= 14]
        # only the pupils (round, 2) are that thick: a brow stroke would make it 3 or 4
        self.assertLessEqual(len(thick), 2)
        # the lid lines of both eyes are still there (the pupils are not: they are drawn by the player)
        self.assertGreaterEqual(ink.sum(), 2 * 1000)


if __name__ == "__main__":
    unittest.main()
