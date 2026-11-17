"""Run every tile through a frozen encoder and save one feature bag per slide.

  data/features/<image_id>.pt = {"features": (N, D) float16, "coords": (N, 2)}
  D = 1024 for resnet50, 768 for hibou-b, 3072 for midnight (see iris/features.py)

Usage:
  python scripts/03_extract_features.py              # CPU (slow, fine for ~300 slides)
  python scripts/03_extract_features.py --device cuda  # on Kaggle / Colab
  python scripts/03_extract_features.py --encoder midnight --out data/features_midnight
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from iris.config import PATHS, TileConfig
from iris.data.tiling import open_slide, read_tile
from iris.features import FEAT_DIMS, build_extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="tiles read at the same time")
    ap.add_argument("--encoder", default="resnet50", choices=list(FEAT_DIMS))
    ap.add_argument("--out", type=Path, default=PATHS.features, help="folder for the feature bags")
    args = ap.parse_args()

    cfg = TileConfig()
    model, tf = build_extractor(args.device, args.encoder)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    pool = ThreadPoolExecutor(args.workers)

    for cpath in tqdm(sorted(PATHS.coords.glob("*.npy"))[: args.limit]):
        out = args.out / f"{cpath.stem}.pt"
        if out.exists():
            continue
        coords = np.load(cpath)
        if len(coords) == 0:
            continue
        slide = open_slide(PATHS.images / f"{cpath.stem}.tiff")
        feats = []
        for i in range(0, len(coords), args.batch):
            batch = torch.stack(list(pool.map(lambda xy: tf(read_tile(slide, xy, cfg)), coords[i : i + args.batch])))
            feats.append(model(batch.to(args.device)).cpu().half())   # fp16 halves disk use
        torch.save({"features": torch.cat(feats), "coords": torch.from_numpy(coords)}, out)

    # TODO (speed): reading tiles is single-threaded here. If it's the bottleneck
    # (check with htop: one core busy, others idle), wrap the tile reading in a
    # torch Dataset + DataLoader(num_workers=4..8).


if __name__ == "__main__":
    main()
