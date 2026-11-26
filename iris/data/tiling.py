"""Cut a slide into a grid of tiles, keeping only tiles that contain tissue.

We do NOT save tile images to disk (that would be hundreds of GB). We save the
tile *coordinates* only, and read pixels on the fly when extracting features.
"""
from pathlib import Path

import numpy as np
import openslide
from PIL import Image

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


# ---------------------------------------------------------------------------------------------
# Tiling by MICRONS, for slides from other datasets (scripts/13_external_features.py).
#
# The PANDA 20x tiles are 256 px at level 0, which is 0.486 um/px (Radboud) or 0.503 um/px
# (Karolinska): about 128 x 128 um of tissue. A slide scanned at 40x (0.25 um/px) needs a
# 512 px square for the same tissue. So for other slides we work in microns: read the square
# that covers TILE_UM of tissue at the sharpest level that is not finer than needed, then shrink
# it to 256 px. Iris then sees the same amount of tissue at the same zoom as in training.
# ---------------------------------------------------------------------------------------------
TARGET_MPP = 0.5            # um per pixel of the tiles Iris sees (PANDA 20x: 0.486 to 0.503)
MASK_MPP = 8.0              # um per pixel of the tissue mask (PANDA: lowest level, about 8)


def slide_mpp(slide: openslide.OpenSlide) -> float:
    """Microns per pixel at level 0, from the slide file. Stops if the file does not say."""
    try:
        return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except (KeyError, ValueError):
        raise ValueError("slide has no microns-per-pixel information: cannot match PANDA's zoom")


class MicronTiler:
    """Tile coordinates and tile reading for one slide, by microns instead of pyramid level."""

    def __init__(self, slide: openslide.OpenSlide, tile_size: int = 256, target_mpp: float = TARGET_MPP,
                 min_tissue: float = 0.25):
        self.slide, self.tile_size, self.min_tissue = slide, tile_size, min_tissue
        self.mpp0 = slide_mpp(slide)
        self.tile_l0 = tile_size * target_mpp / self.mpp0           # tile side in level-0 pixels
        # sharpest level whose pixels are not bigger than target_mpp (never enlarge a tile)
        ok = [lv for lv, d in enumerate(slide.level_downsamples) if self.mpp0 * d <= target_mpp * 1.01]
        self.level = max(ok)
        self.read_px = int(round(self.tile_l0 / slide.level_downsamples[self.level]))

    def coords(self) -> np.ndarray:
        """(N, 2) level-0 corners of the tiles with at least min_tissue tissue (same grid idea as tile_coords)."""
        s = self.slide
        mpps = [self.mpp0 * d for d in s.level_downsamples]
        mask_level = int(np.argmin([abs(np.log(m / MASK_MPP)) for m in mpps]))
        mask = tissue_mask(thumbnail(s, mask_level))
        down_mask = s.level_downsamples[mask_level]
        side = max(1, int(round(self.tile_l0 / down_mask)))
        W0, H0 = s.level_dimensions[0]
        out = []
        for y in np.arange(0, H0 - self.tile_l0 + 1, self.tile_l0):
            for x in np.arange(0, W0 - self.tile_l0 + 1, self.tile_l0):
                my, mx = int(y / down_mask), int(x / down_mask)
                patch = mask[my: my + side, mx: mx + side]
                if patch.size and patch.mean() >= self.min_tissue:
                    out.append((int(round(x)), int(round(y))))
        return np.array(out, dtype=np.int64).reshape(-1, 2)

    def read(self, xy) -> np.ndarray:
        """One RGB tile (tile_size x tile_size) covering tile_l0 level-0 pixels from corner xy."""
        img = self.slide.read_region((int(xy[0]), int(xy[1])), self.level, (self.read_px, self.read_px)).convert("RGB")
        if self.read_px != self.tile_size:
            img = img.resize((self.tile_size, self.tile_size), Image.Resampling.BOX)
        return np.array(img)
