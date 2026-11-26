"""Feature bags for slides from OTHER datasets (external validation), e.g. PAR (Erbil, Iraq).

Same output as 03_extract_features.py, so the PANDA-trained models can grade them unchanged:
  <out>/<slide name>.pt = {"features": (N, D) float16, "coords": (N, 2) level-0 corners, + info}

Differences from 03:
  - tiles are cut by MICRONS, not by pyramid level (iris/data/tiling.py, MicronTiler): every
    tile covers the same ~128 x 128 um of tissue as a PANDA 20x tile and is shrunk to 256 px,
    whatever the scanner's resolution;
  - slides can be URLs: each one is downloaded, turned into features and deleted again, so
    hundreds of GB of slides never need to be stored at once ("streaming").

Nothing here trains anything: external slides are only ever graded.

Usage:
  python scripts/13_external_features.py --slides a.svs b.svs --out data/external/par_grundium
  python scripts/13_external_features.py --list urls.txt --out ... --encoder hibou-b --device cuda
  python scripts/13_external_features.py ... --stain macenko        # stain-normalised version
"""
import argparse
import shutil
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from iris.data import stain
from iris.data.tiling import MicronTiler, open_slide
from iris.features import FEAT_DIMS, build_extractor


def fetch(src: str, tmp: Path) -> tuple[Path, bool]:
    """Local path of a slide; URLs are downloaded to tmp first. Returns (path, downloaded)."""
    if "://" not in src:
        return Path(src), False
    dest = tmp / src.rstrip("/").split("/")[-1]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(src, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            return dest, True
        except OSError as e:
            print(f"  download failed ({e}), retry {attempt + 1}/3", flush=True)
            time.sleep(10)
    raise RuntimeError(f"could not download {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", default=[], help="slide files or URLs")
    ap.add_argument("--list", type=Path, help="text file with one slide path or URL per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder", default="hibou-b", choices=list(FEAT_DIMS))
    ap.add_argument("--stain", choices=["none", "macenko"], default="none")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    sources = list(args.slides)
    if args.list:
        sources += [l.strip() for l in open(args.list) if l.strip() and not l.startswith("#")]
    args.out.mkdir(parents=True, exist_ok=True)
    model, tf = build_extractor(args.device, args.encoder)
    pool = ThreadPoolExecutor(args.workers)
    tmp = Path(tempfile.mkdtemp(prefix="iris_slides_"))

    for n, src in enumerate(sources, 1):
        name = src.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
        out = args.out / f"{name}.pt"
        if out.exists():
            continue
        t0 = time.time()
        path, downloaded = fetch(src, tmp)
        try:
            slide = open_slide(path)
            tiler = MicronTiler(slide)
            coords = tiler.coords()
            bag = {"mpp": tiler.mpp0, "level": tiler.level, "read_px": tiler.read_px}
            if len(coords) == 0:
                print(f"[{n}/{len(sources)}] {name}: no tissue tiles, skipped", flush=True)
                continue
            if args.stain == "macenko":
                he, max_c, bag["stain_reference_colours"] = stain.fit_slide(slide, coords, None, reader=tiler.read)
                load = lambda xy: tf(stain.normalize(tiler.read(xy), he, max_c))
            else:
                load = lambda xy: tf(tiler.read(xy))
            feats = []
            for i in range(0, len(coords), args.batch):
                batch = torch.stack(list(pool.map(load, coords[i: i + args.batch])))
                with torch.no_grad():
                    feats.append(model(batch.to(args.device)).cpu().half())
            bag.update(features=torch.cat(feats), coords=torch.from_numpy(coords))
            torch.save(bag, out)
            print(f"[{n}/{len(sources)}] {name}: {len(coords)} tiles, {tiler.mpp0:.3f} um/px, level {tiler.level}, "
                  f"read {tiler.read_px} px, {time.time() - t0:.0f} s", flush=True)
        finally:
            if downloaded:
                path.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
