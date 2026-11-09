"""Tissue detection: which parts of the slide are tissue and which are glass.

Most of a biopsy slide is empty white background. Tiling it would waste compute
and feed the model garbage, so we first build a binary tissue mask on a tiny
thumbnail of the slide.
"""
import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening, remove_small_objects


def tissue_mask(rgb: np.ndarray, min_object_px: int = 64) -> np.ndarray:
    """Return a boolean mask (H, W), True where there is tissue.

    Trick: stained tissue is *coloured* (pink/purple) while the background is
    white/grey, so the HSV saturation channel separates them well. Otsu picks
    the threshold automatically per slide.
    """
    sat = rgb2hsv(rgb)[..., 1]
    # pure-white padding (common in PANDA) would drag Otsu down, so ignore it
    valid = rgb.mean(axis=-1) < 250
    if valid.sum() < 10:
        return np.zeros(sat.shape, dtype=bool)
    thr = threshold_otsu(sat[valid])
    mask = (sat > thr) & valid

    # clean-up: close small gaps, remove specks of dust / pen marks
    mask = closing(mask, disk(2))
    mask = opening(mask, disk(1))
    mask = remove_small_objects(mask, max_size=min_object_px)

    # TODO (later): PANDA has pen marks (green/blue/black ink) on some slides.
    # They are saturated too, so they pass this filter. Look at a few masks in
    # notebooks/ and decide if you need an extra "ink" filter (hint: hue range).
    return mask
