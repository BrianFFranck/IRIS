"""Cut a slide into a grid of tiles, keeping only tiles that contain tissue.

We do NOT save tile images to disk (that would be hundreds of GB). We save the
tile *coordinates* only, and read pixels on the fly when extracting features.
"""
from pathlib import Path

import numpy as np
import openslide

from iris.config import TileConfig
from iris.data.tissue import tissue_mask


def open_slide(path: Path) -> openslide.OpenSlide:
    return openslide.OpenSlide(str(path))


def thumbnail(slide: openslide.OpenSlide, level: int) -> np.ndarray:
    """Read a whole pyramid level as RGB. Only do this for the small levels!"""
    w, h = slide.level_dimensions[level]
    return np.asarray(slide.read_region((0, 0), level, (w, h)).convert("RGB"))


def tile_coords(slide: openslide.OpenSlide, cfg: TileConfig) -> np.ndarray:
    """Return an (N, 2) int array of tile top-left corners in LEVEL-0 pixels.

    Level-0 coordinates are what openslide.read_region() expects, whatever
    level you read from, so we always store them that way.
    """
    if cfg.level >= slide.level_count:
        raise ValueError(f"slide has {slide.level_count} level(s), can't tile at level {cfg.level}")
    mask_level = cfg.mask_level if cfg.mask_level >= 0 else slide.level_count - 1
    mask = tissue_mask(thumbnail(slide, mask_level))

    down_tile = slide.level_downsamples[cfg.level]      # tile level -> level 0
    down_mask = slide.level_downsamples[mask_level]     # mask level -> level 0
    tile_l0 = cfg.tile_size * down_tile                  # tile side in level-0 px
    tile_in_mask = tile_l0 / down_mask                   # tile side in mask px

    W0, H0 = slide.level_dimensions[0]
    coords = []
    for y in np.arange(0, H0 - tile_l0 + 1, tile_l0):
        for x in np.arange(0, W0 - tile_l0 + 1, tile_l0):
            my, mx = int(y / down_mask), int(x / down_mask)
            s = max(1, int(round(tile_in_mask)))
            patch = mask[my : my + s, mx : mx + s]
            if patch.size and patch.mean() >= cfg.min_tissue:
                coords.append((int(x), int(y)))
    return np.array(coords, dtype=np.int64).reshape(-1, 2)


def read_tile(slide: openslide.OpenSlide, xy, cfg: TileConfig) -> np.ndarray:
    x, y = int(xy[0]), int(xy[1])
    img = slide.read_region((x, y), cfg.level, (cfg.tile_size, cfg.tile_size))
    return np.array(img.convert("RGB"))  # writable copy (avoids a torch warning)
