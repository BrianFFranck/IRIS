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

Usage:
  python scripts/08_cross_centre.py              # all folds (about 1-2 h on the laptop CPU)
  python scripts/08_cross_centre.py --epochs 1 --folds 0    # quick test
"""
import argparse
import json
from datetime import datetime

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
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    cfg.feat_dim = int(torch.load(next(PATHS.features.glob("*.pt")), weights_only=True)["features"].shape[1])
    df = pd.read_csv(PATHS.splits)                      # development slides only: the test set stays locked
    folds = args.folds if args.folds is not None else range(cfg.n_folds)
    run_dir = PATHS.runs / f"{datetime.now():%Y%m%d-%H%M}_crosscentre"

    # 1) train one model per (centre, fold), on that centre only
    for centre in CENTRES:
        own = df[df.data_provider == centre]
        for k in folds:
            out = run_dir / f"train_{centre}" / f"fold{k}"
            train_ds = BagDataset(own[own.fold != k], PATHS.features)
            val_ds = BagDataset(own[own.fold == k], PATHS.features)
            print(f"\n=== train on {centre}, fold {k}: {len(train_ds)} train / {len(val_ds)} val slides")
            model = MODELS["abmil"](feat_dim=cfg.feat_dim, hidden_dim=cfg.hidden_dim,
                                    n_classes=cfg.n_classes, dropout=cfg.dropout)
            train_fold(model, train_ds, val_ds, cfg, out, device=args.device)

    # 2) for each test centre and fold: within vs cross, on the same slides
    results = {"feat_dim": cfg.feat_dim, "epochs": cfg.epochs, "folds": {}}
    for test_c in CENTRES:
        other = [c for c in CENTRES if c != test_c][0]
        pooled = {"yt": [], "within": [], "cross": []}
        for k in folds:
            ds = BagDataset(df[(df.data_provider == test_c) & (df.fold == k)], PATHS.features)
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
