"""Grade external slides with the PANDA-trained fold models (an ensemble), no retraining.

Each of the 5 fold models (trained on PANDA development slides only) gives ISUP probabilities;
the ensemble averages them and takes the most likely grade. External slides are never used for
training or for choosing anything, so the result is an honest external test.

Usage:
  python scripts/14_grade_external.py FEATURES_DIR RUN_DIR --out grades.csv
  e.g. data/external/par_grundium runs_kaggle/trainhibou/runs/20261122-1942_abmil
Output CSV: slide, tiles, ensemble grade, the 6 averaged probabilities, each fold's grade.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from iris.config import TrainConfig
from iris.models import MODELS


def load_models(run_dir, feat_dim, cfg):
    models = []
    for k in range(cfg.n_folds):
        m = MODELS["abmil"](feat_dim=feat_dim, hidden_dim=cfg.hidden_dim, n_classes=cfg.n_classes, dropout=cfg.dropout)
        m.load_state_dict(torch.load(Path(run_dir) / f"fold{k}" / "best.pt", map_location="cpu"))
        models.append(m.eval())
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = TrainConfig()
    files = sorted(args.features.glob("*.pt"))
    feat_dim = torch.load(files[0], weights_only=True)["features"].shape[1]
    models = load_models(args.run_dir, feat_dim, cfg)
    rows = []
    for f in files:
        bag = torch.load(f, weights_only=True)
        x = bag["features"].float()
        with torch.no_grad():
            probs = torch.stack([m(x)["logits"].softmax(-1).squeeze(0) for m in models])   # (folds, 6)
        mean = probs.mean(0)
        rows.append({"slide": f.stem, "tiles": len(x), "grade": int(mean.argmax()),
                     **{f"p{g}": round(float(mean[g]), 4) for g in range(cfg.n_classes)},
                     **{f"fold{k}": int(p.argmax()) for k, p in enumerate(probs)}})
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
