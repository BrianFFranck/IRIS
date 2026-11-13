"""Explainability: paint attention back onto the slide.

Inputs:
  - slide_path: the slide's .tiff
  - coords:     bag["coords"], (N, 2) tile corners in level-0 px
  - attention:  model(feats)["attention"], (N,) one number per tile

Colours show each tile's attention RANK (0 = lowest, 1 = highest), not the raw
weight: raw weights sum to 1, so most of them are tiny and the map would be
unreadable. A rank map shows WHERE Iris looked relative to the rest of the slide,
not HOW concentrated the attention was; report that separately, in numbers.
Glass with no tile stays uncoloured, so tissue that was never tiled is visible too.

Later: on Radboud slides, compare high attention with the Gleason 3/4/5 regions
in PANDA's label masks (train_label_masks/). That turns the picture into a number.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

from iris.config import TileConfig
from iris.data.tiling import open_slide, thumbnail


def attention_heatmap(slide_path, coords, attention, level: int = 2):
    
    slide = open_slide(slide_path)
    img = thumbnail(slide, level)
    down = slide.level_downsamples[level]                    
    size = round(TileConfig().tile_size * 
    slide.level_downsamples[TileConfig().level] / down)
    pos = (np.asarray(coords) // down).astype(int)             
    attention = torch.as_tensor(attention).detach().numpy()

    ranks = rankdata(attention) / len(attention)          
    heat = np.zeros(img.shape[:2])                        

    for k in range(len(attention)):                       
        x, y = pos[k]
        heat[y:y + size, x:x + size] = ranks[k]

    shown = np.ma.masked_equal(heat, 0)                   

    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))                     
    plt.imshow(img)
    plt.imshow(shown, cmap="jet", alpha=0.4)
    plt.colorbar(label="attention rank (0 = lowest, 1 = highest)")
    return fig
