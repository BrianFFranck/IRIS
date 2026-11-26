"""Experiment C: does Iris still work when it is tested on the other centre?

Fair design: every comparison is on the SAME test slides, so only "where the model
learned" changes, not the grade mix (which differs a lot between the two centres).

For each centre and each fold k, train one model on that centre's slides outside fold k
(best epoch chosen on the same centre's fold k, never on the other centre). Then, for
test centre B and fold k:
  within = the model trained on B,          tested on B's fold k
  cross  = the model trained on the other,  tested on the same B fold k
The two hospitals share no patients, so the cross-centre test has no patient leakage.

Output: runs/<date>_crosscentre/ with one folder per (train centre, fold) and
results.json (per-fold QWK, GG2 detection, paired bootstrap 95% CI of the difference).

--norm centre: before training, rescale each centre's tile features to average 0 and spread 1,
using the mean and standard deviation of every tile of that centre's development slides. No
grade labels are used, only the features. Each slide is always rescaled with its own centre's
numbers, so within and cross see exactly the same inputs; only the model changes.
Question: is the gap between centres a simple shift of the features?

Usage:
  python scripts/08_cross_centre.py              # all folds (about 1-2 h on the laptop CPU)
  python scripts/08_cross_centre.py --epochs 1 --folds 0    # quick test
  python scripts/08_cross_centre.py --norm centre           # with per-centre feature rescaling
  python scripts/08_cross_centre.py --aug-features data/features_aug   # with stain augmentation (training slides only)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iris.config import PATHS, TrainConfig
from iris.data.dataset import BagDataset
from iris.metrics import qwk
from iris.models import MODELS
from iris.train import run_epoch, train_fold

CENTRES = ["radboud", "karolinska"]


def predict(model_path, ds, cfg, device):
    model = MODELS["abmil"](feat_dim=cfg.feat_dim, hidden_dim=cfg.hidden_dim,
                            n_classes=cfg.n_classes, dropout=cfg.dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    _, _, yt, yp = run_epoch(model, dl, nn.CrossEntropyLoss(), None, device)
    return list(map(int, yt)), list(map(int, yp))


def centre_stats(df, features_dir):
    """Mean and standard deviation of every feature, over all tiles of each centre's slides."""
    stats = {}
    for centre in CENTRES:
        total, total_sq, n = 0.0, 0.0, 0
        for image_id in df[df.data_provider == centre].image_id:
            path = features_dir / f"{image_id}.pt"
            if not path.exists():
                continue
            f = torch.load(path, weights_only=True)["features"].double()
            total, total_sq, n = total + f.sum(0), total_sq + (f ** 2).sum(0), n + len(f)
        mean = total / n
        std = (total_sq / n - mean ** 2).clamp_min(1e-12).sqrt()
        stats[centre] = (mean.float(), std.float())
        print(f"{centre}: {n} tiles, mean of means {mean.mean():.3f}, mean of stds {std.mean():.3f}")
    return stats


def boundary_counts(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    return {"gg2_total": int((yt == 2).sum()), "gg2_called_gg2": int(((yt == 2) & (yp == 2)).sum()),
            "gg1_called_gg2": int(((yt == 1) & (yp == 2)).sum())}


def paired_bootstrap(yt, yp_within, yp_cross, n=2000, seed=42):
    """95% CI of QWK(cross) - QWK(within) on the same slides, resampling slides."""
    rng = np.random.default_rng(seed)
    yt, a, b = map(np.array, (yt, yp_within, yp_cross))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        diffs.append(qwk(yt[i], b[i]) - qwk(yt[i], a[i]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--norm", choices=["none", "centre"], default="none")
    ap.add_argument("--aug-features", type=Path, default=None,
                    help="stain augmentation: folder of recoloured bags; used for TRAINING slides only")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    cfg.feat_dim = int(torch.load(next(PATHS.features.glob("*.pt")), weights_only=True)["features"].shape[1])
    df = pd.read_csv(PATHS.splits)                      # development slides only: the test set stays locked
    folds = args.folds if args.folds is not None else range(cfg.n_folds)
    suffix = ("_norm" if args.norm == "centre" else "") + ("_aug" if args.aug_features else "")
    run_dir = PATHS.runs / f"{datetime.now():%Y%m%d-%H%M}_crosscentre{suffix}"
    norm = centre_stats(df, PATHS.features) if args.norm == "centre" else None

    # 1) train one model per (centre, fold), on that centre only
    for centre in CENTRES:
        own = df[df.data_provider == centre]
        for k in folds:
            out = run_dir / f"train_{centre}" / f"fold{k}"
            train_ds = BagDataset(own[own.fold != k], PATHS.features, norm=norm, aug_dir=args.aug_features)
            val_ds = BagDataset(own[own.fold == k], PATHS.features, norm=norm)
            print(f"\n=== train on {centre}, fold {k}: {len(train_ds)} train / {len(val_ds)} val slides")
            model = MODELS["abmil"](feat_dim=cfg.feat_dim, hidden_dim=cfg.hidden_dim,
                                    n_classes=cfg.n_classes, dropout=cfg.dropout)
            train_fold(model, train_ds, val_ds, cfg, out, device=args.device)

    # 2) for each test centre and fold: within vs cross, on the same slides
    results = {"feat_dim": cfg.feat_dim, "epochs": cfg.epochs, "norm": args.norm,
               "stain_augmentation": bool(args.aug_features), "folds": {}}
    for test_c in CENTRES:
        other = [c for c in CENTRES if c != test_c][0]
        pooled = {"yt": [], "within": [], "cross": []}
        for k in folds:
            ds = BagDataset(df[(df.data_provider == test_c) & (df.fold == k)], PATHS.features, norm=norm)
            yt, yp_w = predict(run_dir / f"train_{test_c}" / f"fold{k}" / "best.pt", ds, cfg, args.device)
            yt2, yp_c = predict(run_dir / f"train_{other}" / f"fold{k}" / "best.pt", ds, cfg, args.device)
            assert yt == yt2
            results["folds"][f"test_{test_c}_fold{k}"] = {
                "slides": len(yt), "qwk_within": qwk(yt, yp_w), "qwk_cross": qwk(yt, yp_c),
                "within": boundary_counts(yt, yp_w), "cross": boundary_counts(yt, yp_c)}
            pooled["yt"] += yt; pooled["within"] += yp_w; pooled["cross"] += yp_c
        results[f"test_{test_c}"] = {
            "slides": len(pooled["yt"]),
            "qwk_within": qwk(pooled["yt"], pooled["within"]), "qwk_cross": qwk(pooled["yt"], pooled["cross"]),
            "diff_ci95": paired_bootstrap(pooled["yt"], pooled["within"], pooled["cross"]),
            "within": boundary_counts(pooled["yt"], pooled["within"]),
            "cross": boundary_counts(pooled["yt"], pooled["cross"])}
        json.dump(pooled, open(run_dir / f"preds_test_{test_c}.json", "w"))

    json.dump(results, open(run_dir / "results.json", "w"), indent=1)
    for test_c in CENTRES:
        r = results[f"test_{test_c}"]
        print(f"\ntest on {test_c} ({r['slides']} slides): within {r['qwk_within']:.3f} | "
              f"cross {r['qwk_cross']:.3f} | diff 95% CI {np.round(r['diff_ci95'], 3).tolist()}")
        print(f"  GG2 caught: within {r['within']['gg2_called_gg2']}/{r['within']['gg2_total']}, "
              f"cross {r['cross']['gg2_called_gg2']}/{r['cross']['gg2_total']}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
