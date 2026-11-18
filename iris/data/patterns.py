"""Tile-level Gleason pattern labels from the Radboud label masks.

Radboud masks give every pixel a number: 0 background, 1 stroma, 2 benign glands,
3/4/5 Gleason pattern 3/4/5 (read from the first colour channel). Karolinska masks only
say cancer / not cancer, so they cannot give pattern labels.

Each tile gets ONE label: the most serious pattern that covers at least PATTERN_MIN of it,
or 0 if none does. A small area of pattern 4 matters to a pathologist, so it is not
outvoted by the pattern 3 or the stroma around it. 5% is the same threshold as the
attention check in notebooks/01_explore.ipynb (fixed 2026-11-13, before looking).
"""
import numpy as np

from iris.config import PATHS, TileConfig
from iris.data.tiling import open_slide, read_tile

PATTERN_MIN = 0.05   # a pattern counts if it covers at least 5% of the tile


def tile_label(mask_tile):
    """One Radboud mask tile -> 0 (no cancer), 3, 4 or 5:
    the most serious Gleason pattern that covers at least PATTERN_MIN of the tile."""
    for pattern in (5, 4, 3):          # most serious first, so 8% pattern 4 beats 25% pattern 3
        if (mask_tile == pattern).mean() >= PATTERN_MIN:
            return pattern
    return 0


def slide_tile_labels(image_id: str, coords: np.ndarray, cfg: TileConfig = TileConfig()) -> np.ndarray:
    """Labels for all tiles of one Radboud slide, in the same order as its coords."""
    mask = open_slide(PATHS.masks / f"{image_id}_mask.tiff")
    return np.array([tile_label(read_tile(mask, xy, cfg)[:, :, 0]) for xy in coords], dtype=np.int8)
