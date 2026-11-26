"""Stain normalisation (Macenko et al. 2009): repaint every slide with the same H&E colours.

H&E slides are coloured by two stains: haematoxylin (purple-blue, nuclei) and eosin (pink,
cytoplasm and stroma). How strong and which exact shade each stain has depends on the lab and
the scanner, which is why Karolinska looks purple and Radboud pale pink.

Macenko's idea, in four steps:
  1. Turn colours into optical density (OD): how much light the tissue absorbs. In OD, a
     pixel is (amount of haematoxylin) x (haematoxylin colour) + (amount of eosin) x (eosin colour).
  2. Find this slide's two stain colours: the two "edges" of its tissue pixels in OD space.
  3. Work out, for every pixel, how much of each stain it has.
  4. Repaint: the same amounts, but with the REFERENCE stain colours and strengths.

The stain colours are estimated once per slide (from its small thumbnail, which is stable),
then applied to every tile of that slide. Estimating per tile would be unstable on tiles
with little tissue.
"""
import numpy as np

IO = 240                                   # brightness of the light through an empty slide
BETA = 0.15                                # ignore nearly transparent pixels (background)
ALPHA = 1                                  # percentile used to find the stain "edges"
# Reference stain colours (columns: haematoxylin, eosin) and strengths, the values used by
# the common open-source Macenko implementations.
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])


def optical_density(rgb):
    return -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1) / IO)


def fit(rgb):
    """Stain colours (3x2) and strengths (2,) of a slide, from an RGB image of it (e.g. a thumbnail)."""
    od = optical_density(rgb)
    od = od[(od > BETA).all(axis=1)]                     # tissue pixels only
    if len(od) < 100:
        raise ValueError("too little tissue to estimate the stains")
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, 1:3]                                 # the 2 directions with the most spread
    proj = od @ plane
    angle = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(angle, [ALPHA, 100 - ALPHA])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T   # haematoxylin first
    he = he * np.sign(he.sum(axis=0))                    # stain colours are positive in OD
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.percentile(conc, 99, axis=1)
    return he, max_c


def normalize(tile, he, max_c):
    """Repaint one RGB tile (H, W, 3) with the reference stains, given its slide's fit().

    One matrix does steps 3 and 4 at once: OD -> stain amounts (pseudo-inverse of the slide's
    colours) -> rescaled to the reference strengths -> OD with the reference colours.
    """
    h, w, _ = tile.shape
    m = (HE_REF * (MAXC_REF / max_c)) @ np.linalg.pinv(he)            # 3x3
    od = -np.log((tile.reshape(-1, 3).astype(np.float32) + 1) / IO)
    out = IO * np.exp(-(od @ m.T.astype(np.float32)))
    return np.clip(out, 0, 255).reshape(h, w, 3).astype(np.uint8)



MIN_COS = 0.95      # a slide's stain colours must be this close to normal H&E, else use the reference colours


def _cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def fit_slide(slide, coords, cfg, n_tiles=16):
    """fit() on up to n_tiles of the slide's own tiles (already tissue), spread over the slide.

    Pen ink is removed first (see tissue.py). Safeguard: on very pale slides Macenko cannot find
    the haematoxylin colour and returns a wrong one; if either colour is far from normal H&E
    (cosine below MIN_COS), the reference colours are kept and only the strengths are fitted.
    Returns (stain colours, strengths, True if the reference colours were used).
    """
    from iris.data.tiling import read_tile
    from iris.data.tissue import ink_mask
    pick = coords[np.linspace(0, len(coords) - 1, min(n_tiles, len(coords))).astype(int)]
    px = np.concatenate([t[~ink_mask(t)] for t in (read_tile(slide, xy, cfg) for xy in pick)])
    he, max_c = fit(px[:, None, :])
    if min(_cos(he[:, 0], HE_REF[:, 0]), _cos(he[:, 1], HE_REF[:, 1])) >= MIN_COS:
        return he, max_c, False
    od = optical_density(px)
    od = od[(od > BETA).all(axis=1)]
    conc = np.linalg.lstsq(HE_REF, od.T, rcond=None)[0]
    return HE_REF, np.percentile(conc, 99, axis=1), True


# ---------------------------------------------------------------------------------------------
# Stain AUGMENTATION (after Tellez et al. 2018, "HED jitter"): instead of repainting every slide
# to the same colours, give each slide a random but realistic new colouring, so the model learns
# that the exact shade does not matter, while each slide's own shading is kept, not erased.
#
#   1. RGB -> optical density -> amounts of haematoxylin, eosin and a third "rest" channel
#      (fixed stain colours of Ruifrok & Johnston 2001, the same for every slide)
#   2. the haematoxylin and eosin amounts are multiplied by random factors
#   3. back to RGB
# Only multiplying (no shift) keeps empty glass white. The same factors are used for every tile
# of a slide, so a whole slide looks as if it had been stained in another lab.
#
# How big: measured on 150 local slides per centre (2026-11-26), the median Karolinska slide has
# 1.15x the haematoxylin and 2.0x the eosin of the median Radboud slide, and within Radboud the
# 10th-90th percentile spread is about 0.8x (H) and 0.7x (E) of the median. So the factors go up
# to 2x and down to 1/2 for eosin, and 1.4x and 1/1.4 for haematoxylin, spread evenly on a log
# scale (doubling is as likely as halving).
# ---------------------------------------------------------------------------------------------
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],     # haematoxylin
                         [0.07, 0.99, 0.11],     # eosin
                         [0.27, 0.57, 0.78]])    # rest (DAB in the original paper)
RGB_FROM_HED = RGB_FROM_HED / np.linalg.norm(RGB_FROM_HED, axis=1, keepdims=True)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
MAX_FACTOR = (1.4, 2.0)                          # haematoxylin, eosin


def hed_amounts(rgb):
    """Amounts of haematoxylin, eosin and rest per pixel, shape (pixels, 3)."""
    return optical_density(rgb) @ HED_FROM_RGB


def random_factors(rng, max_factor=MAX_FACTOR):
    """One slide's random colouring: a factor per stain, log-uniform in [1/max, max]; rest = 1."""
    f = np.ones(3)
    f[:2] = np.exp(rng.uniform(-np.log(max_factor), np.log(max_factor)))
    return f


def jitter(tile, factors):
    """Recolour one RGB tile (H, W, 3): stain amounts x factors, then back to RGB.

    One 3x3 matrix does it all (about 1 ms per 256 x 256 tile). factors (1, 1, 1) = no change.
    """
    h, w, _ = tile.shape
    m = ((HED_FROM_RGB * factors) @ RGB_FROM_HED).astype(np.float32)
    od = -np.log((tile.reshape(-1, 3).astype(np.float32) + 1) / IO)
    out = IO * np.exp(-(od @ m)) - 1                                   # -1 undoes the +1 above
    return np.clip(np.rint(out), 0, 255).reshape(h, w, 3).astype(np.uint8)
