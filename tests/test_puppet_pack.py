"""Tests for rigstudio/puppet_job.py (the rig-puppet-pack workflow's logic), run without ComfyUI:

    python -m unittest tests.test_puppet_pack -v

The end-to-end test needs the four sheets and the finished puppet from Rig Studio's samples
(RIG_SAMPLES, default D:/rig-studio/samples/puppet/bluebob2) and is skipped without them.
"""
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


if __name__ == "__main__":
    unittest.main()
