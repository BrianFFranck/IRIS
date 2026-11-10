"""Tissue detection: which parts of the slide are tissue and which are glass.

Most of a biopsy slide is empty white background. Tiling it would waste compute
and feed the model garbage, so we first build a binary tissue mask on a tiny
thumbnail of the slide.
"""
import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening, remove_small_objects

# H&E tissue is pink/purple. Pathologists mark slides with green, blue or black
# pen, and that ink is strongly coloured too, so it must be removed explicitly.
INK_HUE = (0.15, 0.70)     # hue range of green/blue ink (H&E sits outside it)
INK_MIN_SAT = 0.12         # ...when it is clearly coloured
BLACK_MAX_VAL = 0.25       # very dark...
BLACK_MAX_SAT = 0.35       # ...and grey = black ink (dark tumour is dark PURPLE, so it stays)
MAX_THRESHOLD = 0.08       # faintly coloured is enough to count as tissue


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    """True where there is pen ink (green, blue or black)."""
    hsv = rgb2hsv(rgb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    coloured_ink = (hue > INK_HUE[0]) & (hue < INK_HUE[1]) & (sat > INK_MIN_SAT)
    black_ink = (val < BLACK_MAX_VAL) & (sat < BLACK_MAX_SAT)
    return coloured_ink | black_ink


def tissue_mask(rgb: np.ndarray, min_object_px: int = 64) -> np.ndarray:
    """Return a boolean mask (H, W), True where there is tissue.

    Trick: stained tissue is *coloured* (pink/purple) while the background is
    white/grey, so the HSV saturation channel separates them well. Otsu picks
    the threshold automatically per slide.

    Two guards, both found on real PANDA slides (e.g. 1364c10e, ISUP 5):
      - pen ink is removed first, otherwise Otsu splits "ink vs everything"
        and pale tissue is thrown away with the glass;
      - the threshold is capped, so a pale slide still counts as tissue.
    """
    sat = rgb2hsv(rgb)[..., 1]
    # pure-white padding (common in PANDA) would drag Otsu down, so ignore it
    valid = (rgb.mean(axis=-1) < 250) & ~ink_mask(rgb)
    if valid.sum() < 10:
        return np.zeros(sat.shape, dtype=bool)
    thr = min(threshold_otsu(sat[valid]), MAX_THRESHOLD)
    mask = (sat > thr) & valid

    # clean-up: close small gaps, remove specks of dust
    mask = closing(mask, disk(2))
    mask = opening(mask, disk(1))
    mask = remove_small_objects(mask, max_size=min_object_px)
    return mask
