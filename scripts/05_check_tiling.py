"""Measure how good the tiling is, using PANDA's pixel-level label masks as the answer key.

For every slide that has a mask, compare the kept tiles with the labelled tissue:

  tissue_recall   share of the real tissue that falls inside a kept tile
                  (low = tissue thrown away = false negatives)
  tile_precision  share of the kept tiles' area that is real tissue
                  (low = tiles full of glass / ink = false positives)
  cancer_recall   share of the CANCER that falls inside a kept tile
                  (the number that matters most: missed cancer can never be found)

Mask labels (red channel):
  Radboud:     0 background, 1 stroma, 2 benign epithelium, 3/4/5 Gleason pattern
  Karolinska:  0 background, 1 benign tissue, 2 cancer
Karolinska masks were made semi-automatically and are rougher than Radboud's.

Usage:
  python scripts/01_download_subset.py --n 300 --masks   # once, to get the masks
  python scripts/05_check_tiling.py
"""
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

from iris.config import PATHS, TileConfig


def label_map(image_id: str) -> np.ndarray:
    """The mask at its lowest resolution, as one label per pixel."""
    m = openslide.OpenSlide(str(PATHS.masks / f"{image_id}_mask.tiff"))
    lvl = m.level_count - 1
    rgb = np.asarray(m.read_region((0, 0), lvl, m.level_dimensions[lvl]).convert("RGB"))
    return rgb[..., 0], m.level_downsamples[lvl]


def covered_map(shape, coords, down_mask: float, tile_l0: float) -> np.ndarray:
    """True on every mask pixel that lies inside a kept tile."""
    cov = np.zeros(shape, dtype=bool)
    s = max(1, int(round(tile_l0 / down_mask)))
    for x, y in coords:
        cx, cy = int(x / down_mask), int(y / down_mask)
        cov[cy : cy + s, cx : cx + s] = True
    return cov


def check_slide(image_id: str, provider: str, cfg: TileConfig) -> dict:
    labels, down = label_map(image_id)
    coords = np.load(PATHS.coords / f"{image_id}.npy")
    slide = openslide.OpenSlide(str(PATHS.images / f"{image_id}.tiff"))
    tile_l0 = cfg.tile_size * slide.level_downsamples[cfg.level]
    cov = covered_map(labels.shape, coords, down, tile_l0)

    tissue = labels > 0
    cancer = labels >= 3 if provider == "radboud" else labels == 2
    ratio = lambda a, b: a.sum() / b.sum() if b.sum() else np.nan
    return {
        "image_id": image_id,
        "tiles": len(coords),
        "tissue_recall": ratio(tissue & cov, tissue),
        "tile_precision": ratio(tissue & cov, cov),
        "cancer_recall": ratio(cancer & cov, cancer),
        "cancer_px": int(cancer.sum()),
    }


def main():
    cfg = TileConfig()
    df = pd.read_csv(PATHS.train_csv).set_index("image_id")
    ids = sorted(p.name.removesuffix("_mask.tiff") for p in PATHS.masks.glob("*_mask.tiff"))
    ids = [i for i in ids if (PATHS.coords / f"{i}.npy").exists()]
    print(f"checking {len(ids)} slides that have both tiles and a label mask")

    rows = [check_slide(i, df.loc[i, "data_provider"], cfg) for i in tqdm(ids)]
    res = pd.DataFrame(rows).join(df[["data_provider", "isup_grade"]], on="image_id")
    res.to_csv(PATHS.data / "tiling_check.csv", index=False)

    cols = ["tissue_recall", "tile_precision", "cancer_recall"]
    print("\nmedian per lab (1.00 = perfect):")
    print(res.groupby("data_provider")[cols].median().round(3))
    print("\nworst 10% of slides (10th percentile):")
    print(res.groupby("data_provider")[cols].quantile(0.10).round(3))
    print("\nslides losing the most cancer:")
    worst = res[res.cancer_px > 0].nsmallest(5, "cancer_recall")
    print(worst[["image_id", "data_provider", "isup_grade", "tiles", "cancer_recall"]].round(3).to_string(index=False))
    print(f"\nfull table saved to {PATHS.data / 'tiling_check.csv'}")


if __name__ == "__main__":
    main()
