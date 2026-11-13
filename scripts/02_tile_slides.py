"""For every slide: tissue mask -> tile grid -> save coordinates.

Usage:
  python scripts/02_tile_slides.py            # all slides in data/panda/train_images
  python scripts/02_tile_slides.py --limit 5  # quick try
  python scripts/02_tile_slides.py --ids ids.txt   # only the slides listed (one image_id per line)
"""
import argparse

import numpy as np
from tqdm import tqdm

from iris.config import PATHS, TileConfig
from iris.data.tiling import open_slide, tile_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="text file with one image_id per line: tile only those slides")
    args = ap.parse_args()

    cfg = TileConfig()
    PATHS.coords.mkdir(parents=True, exist_ok=True)
    slides = sorted(PATHS.images.glob("*.tiff"))
    if args.ids:
        wanted = {line.strip() for line in open(args.ids) if line.strip()}
        slides = [p for p in slides if p.stem in wanted]
        print(f"tiling {len(slides)} of the {len(wanted)} listed slides")
    slides = slides[: args.limit]
    counts = []
    for path in tqdm(slides):
        out = PATHS.coords / f"{path.stem}.npy"
        if out.exists():
            continue
        coords = tile_coords(open_slide(path), cfg)
        np.save(out, coords)
        counts.append(len(coords))
        if len(coords) == 0:
            tqdm.write(f"WARNING {path.stem}: 0 tiles (blank slide? mask failed?)")

    if counts:
        print(f"tiles per slide: min {min(counts)}, median {int(np.median(counts))}, max {max(counts)}")
    # TODO: open a notebook and DRAW the tile grid on top of 5-10 thumbnails.
    # Do not trust this step until you've seen it with your own eyes.


if __name__ == "__main__":
    main()
