"""Give every tile of the Radboud development slides a Gleason pattern label (0, 3, 4 or 5).

  data/tile_labels/<image_id>.npy = one label per tile, same order as data/coords/<image_id>.npy

Only Radboud development slides with a label mask: Karolinska masks have no patterns,
and the held-out test slides are never opened. See iris/data/patterns.py for the rule.

Usage:
  python scripts/07_tile_labels.py
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from iris.config import PATHS
from iris.data.patterns import slide_tile_labels


def main():
    dev = pd.read_csv(PATHS.splits)          # development slides only: the test set stays locked
    slides = dev[dev.data_provider == "radboud"].image_id
    slides = [s for s in slides
              if (PATHS.masks / f"{s}_mask.tiff").exists() and (PATHS.coords / f"{s}.npy").exists()]
    out_dir = PATHS.data / "tile_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(slides)} Radboud development slides with a mask and tiles -> {out_dir}")

    for image_id in tqdm(slides):
        out = out_dir / f"{image_id}.npy"
        if out.exists():
            continue
        coords = np.load(PATHS.coords / f"{image_id}.npy")
        np.save(out, slide_tile_labels(image_id, coords))


if __name__ == "__main__":
    main()
