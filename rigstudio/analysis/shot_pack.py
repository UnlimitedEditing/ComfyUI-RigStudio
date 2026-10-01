"""Shot packages: a rigstudio.shot plus its portable STAGE (props, cast, sentences with word timings) in one M3DS PNG.

The stage makes the package self-contained for blocking: the Graydient `rig-block` node compiles a blocking script
against it without the set files or direction sheets. Asset URLs inside the shot (set, puppets, tracks, audio) are
left as they are -- the package carries blocking data, not the media.

  python analysis/shot_pack.py pack samples/shots/kitchen_intro.shot.json      -> samples/shots/kitchen_intro.shotpkg.png
  python analysis/shot_pack.py card samples/shots/kitchen_intro.shotpkg.png    # print the stage card
  python analysis/shot_pack.py unpack x.shotpkg.png                            -> x.shot.json
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3ds

FORMAT, NAME = "rigstudio.shotpkg", "shot.rigstudio.json"


def make(shot, stage):
    return {"format": FORMAT, "version": 1, "shot": shot, "stage": stage}


def to_png(pkg):
    return m3ds.encode(json.dumps(pkg, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), NAME)


def loads(data):
    """PNG bytes (M3DS) or JSON bytes -> package dict."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        data, _ = m3ds.decode_png_bytes(data)
    pkg = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)
    if pkg.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} (format={pkg.get('format')!r})")
    return pkg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack", "card", "unpack"])
    ap.add_argument("path")
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Path(a.path)
    if a.cmd == "pack":
        import blocking as B
        shot = json.loads(p.read_text(encoding="utf-8"))
        pkg = make(shot, B.portable_stage(shot))
        out = Path(a.out) if a.out else p.with_name(p.name.replace(".shot.json", ".shotpkg.png"))
        to_png(pkg).save(out)
        print(f"{out}  ({len(pkg['stage']['props'])} props, {len(pkg['stage']['sentences'])} sentences)")
    else:
        pkg = loads(p.read_bytes())
        if a.cmd == "card":
            import blocking as B
            print(B.stage_card(pkg["stage"], pkg["shot"].get("brief", "")))
        else:
            out = Path(a.out) if a.out else p.with_name(p.name.replace(".shotpkg.png", ".shot.json"))
            out.write_text(json.dumps(pkg["shot"], indent=2), encoding="utf-8")
            print(out)


if __name__ == "__main__":
    main()
