"""Run every tile through frozen ResNet-50 and save one feature bag per slide.

  data/features/<image_id>.pt = {"features": (N, 1024) float16, "coords": (N, 2)}

Usage:
  python scripts/03_extract_features.py              # CPU (slow, fine for ~300 slides)
  python scripts/03_extract_features.py --device cuda  # on Kaggle / Colab
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from iris.config import PATHS, TileConfig
from iris.data.tiling import open_slide, read_tile
from iris.features import build_extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = TileConfig()
    model, tf = build_extractor(args.device)
    PATHS.features.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    for cpath in tqdm(sorted(PATHS.coords.glob("*.npy"))[: args.limit]):
        out = PATHS.features / f"{cpath.stem}.pt"
        if out.exists():
            continue
        coords = np.load(cpath)
        if len(coords) == 0:
            continue
        slide = open_slide(PATHS.images / f"{cpath.stem}.tiff")
        feats = []
        for i in range(0, len(coords), args.batch):
            batch = torch.stack([tf(read_tile(slide, xy, cfg)) for xy in coords[i : i + args.batch]])
            feats.append(model(batch.to(args.device)).cpu().half())   # fp16 halves disk use
        torch.save({"features": torch.cat(feats), "coords": torch.from_numpy(coords)}, out)

    # TODO (speed): reading tiles is single-threaded here. If it's the bottleneck
    # (check with htop: one core busy, others idle), wrap the tile reading in a
    # torch Dataset + DataLoader(num_workers=4..8).


if __name__ == "__main__":
    main()
