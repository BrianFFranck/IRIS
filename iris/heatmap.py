"""Explainability: paint attention back onto the slide.

>>> STUB: do this after Iris v1 trains. <<<
  - coords:    data/coords/<id>.npy   (N, 2) tile corners in level-0 px
  - attention: model(feats)["attention"]  (N,)  one number per tile
  - a thumbnail of the slide: iris.data.tiling.thumbnail(slide, level=2)

Plan:
  1. Make an empty float array the size of the thumbnail.
  2. For each tile k: convert coords[k] to thumbnail px (divide by the
     thumbnail level's downsample) and fill that square with attention[k].
  3. Normalise. Raw attention sums to 1, so with 300 tiles the values are tiny.
     Use percentile ranks (scipy.stats.rankdata / N) so the map is readable.
  4. Show the thumbnail with plt.imshow, then the heatmap on top with
     cmap="jet", alpha=0.4, and transparency where there is no tissue.
  5. Sanity check on RADBOUD slides: PANDA has pixel masks
     (train_label_masks/) with Gleason 3/4/5 regions. Does high attention
     overlap the cancer? That is a real, quantitative explainability result.
"""


def attention_heatmap(slide_path, coords, attention, level: int = 2):
    raise NotImplementedError
