"""Attention check at 20x: does the model's attention fall on Gleason pattern 4?

Same measure as the 5x checks in notebooks/01_explore.ipynb (2026-11-13 and 2026-11-17), now on
the 20x tiles and for several encoders at once, on the same slides and the same tiles:

  - slides: Radboud development slides with ISUP 2 (Gleason 3+4) that have a label mask
  - each slide is scored by the fold model that never trained on it (its own fold)
  - a tile "contains pattern 4" if more than 5% of it is pattern 4 in the mask
  - enrichment = share of attention on those tiles / share of tiles that are those tiles
    (1 = no better than looking everywhere equally; slides with no pattern 4 tile are skipped)

Only development slides are used: the test set stays locked.

Usage (on Kaggle, see kaggle/attention20x):
  python scripts/11_attention_check.py --encoder resnet FEATURES_DIR MODELS_DIR \
                                       --encoder hibou FEATURES_DIR MODELS_DIR
  MODELS_DIR holds fold0/best.pt .. fold4/best.pt. Output: attention_check.csv in --out.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from iris.config import PATHS, TileConfig
from iris.data.tiling import open_slide, read_tile
from iris.models import MODELS

PATTERN4_MIN = 0.05           # same threshold as the 5x checks


def load_models(models_dir, feat_dim):
    models = {}
    for k in range(5):
        m = MODELS["abmil"](feat_dim=feat_dim, hidden_dim=256, n_classes=6, dropout=0.25)
        m.load_state_dict(torch.load(Path(models_dir) / f"fold{k}" / "best.pt", map_location="cpu"))
        models[k] = m.eval()
    return models


def summary(res, enc):
    ok = res.dropna(subset=[f"enrich_{enc}"])
    e = ok[f"enrich_{enc}"]
    right = ok[ok[f"pred_{enc}"] == 2][f"enrich_{enc}"]
    wrong = ok[ok[f"pred_{enc}"] != 2][f"enrich_{enc}"]
    print(f"{enc}: {len(ok)} slides measured | median enrichment {e.median():.2f} "
          f"(25th-75th percentile {e.quantile(0.25):.2f}-{e.quantile(0.75):.2f}) | below 1: "
          f"{(e < 1).sum()} ({(e < 1).mean():.0%}) | called GG2 right {right.median():.2f} (n={len(right)}) "
          f"vs wrong {wrong.median():.2f} (n={len(wrong)}) | attention on cancer tiles, median "
          f"{ok[f'cancer_share_{enc}'].median():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", nargs=3, action="append", metavar=("NAME", "FEATURES", "MODELS"), required=True)
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    cfg = TileConfig()

    df = pd.read_csv(PATHS.splits)             # development slides only
    slides = df[(df.data_provider == "radboud") & (df.isup_grade == 2)]
    slides = slides[[(PATHS.masks / f"{i}_mask.tiff").exists() and
                     all((Path(f) / f"{i}.pt").exists() for _, f, _ in args.encoder) for i in slides.image_id]]
    print(f"{len(slides)} Radboud GG2 development slides with a mask and features (tile level {cfg.level})")

    models = {}
    for name, feats, mdir in args.encoder:
        dim = torch.load(next(Path(feats).glob("*.pt")), weights_only=True)["features"].shape[1]
        models[name] = load_models(mdir, dim)

    rows = []
    for _, row in tqdm(slides.iterrows(), total=len(slides)):
        bags = {name: torch.load(Path(f) / f"{row.image_id}.pt", weights_only=True) for name, f, _ in args.encoder}
        coords = next(iter(bags.values()))["coords"]
        assert all(torch.equal(b["coords"], coords) for b in bags.values()), "encoders saw different tiles"

        mask = open_slide(PATHS.masks / f"{row.image_id}_mask.tiff")
        tiles = [read_tile(mask, xy, cfg)[:, :, 0] for xy in coords.numpy()]
        has4 = np.array([(t == 4).mean() > PATTERN4_MIN for t in tiles])
        cancer = np.array([(t >= 3).mean() > PATTERN4_MIN for t in tiles])
        r = {"image_id": row.image_id, "fold": row.fold, "tiles": len(tiles),
             "p4_tiles": int(has4.sum()), "cancer_tiles": int(cancer.sum())}
        for name in bags:
            with torch.no_grad():
                out = models[name][row.fold](bags[name]["features"].float())
            att = out["attention"].numpy()
            r[f"pred_{name}"] = int(out["logits"].argmax())
            r[f"share_{name}"] = float(att[has4].sum())
            r[f"cancer_share_{name}"] = float(att[cancer].sum())
            r[f"enrich_{name}"] = float(att[has4].sum() / has4.mean()) if has4.any() else np.nan
        rows.append(r)

    res = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "attention_check.csv", index=False)
    print(f"\n{(res.p4_tiles == 0).sum()} slides have no tile with pattern 4 in the mask (not measurable)")
    for name, _, _ in args.encoder:
        summary(res, name)
    print(f"saved {args.out / 'attention_check.csv'}")


if __name__ == "__main__":
    main()
