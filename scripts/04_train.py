"""Train a MIL model on the saved feature bags with cross-validation.

Usage:
  python scripts/04_train.py --model meanmax            # the baseline
  python scripts/04_train.py --model abmil              # Iris v1 (after you finish abmil.py)
  python scripts/04_train.py --model meanmax --folds 0  # just one fold, quick
  python scripts/04_train.py --model abmil --class-weights   # rarer grades cost more

A full run (all folds) is written to experiments.md by itself. If it is a new best it is
also pushed and shown on the GitHub profile card; otherwise it stays local. Quick runs
with --folds are never logged. Use --no-log to skip logging altogether.
  python scripts/04_train.py --model abmil --note "what you learned from this run"
"""
import argparse
from pathlib import Path
from datetime import datetime

import torch

import numpy as np
import pandas as pd

from iris.config import PATHS, TrainConfig
from iris.data.dataset import BagDataset
from iris.data.splits import make_folds, make_test_split
from iris.models import MODELS
from iris.runlog import log_run
from iris.train import train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="meanmax")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="which folds to run (default: all)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--note", default="", help="your finding for experiments.md (default: written from the numbers)")
    ap.add_argument("--settings", default="", help="extra settings to record, e.g. 'boundary-weighted loss'")
    ap.add_argument("--no-log", action="store_true", help="don't write this run to experiments.md or the profile")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (model start weights, slide order, dropout); the folds stay the same. Default: 42 as always")
    ap.add_argument("--aug-features", type=Path, default=None,
                    help="stain augmentation: folder of recoloured bags (03_extract_features.py --stain jitter); training only")
    ap.add_argument("--aug-p", type=float, default=0.5, help="chance that a training slide is read recoloured")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--class-weights", action="store_true", help="weight the loss so rarer grades cost more")
    ap.add_argument("--boundary-weight", type=float, default=0.0, help="extra loss for probability across the GG1/GG2 boundary, e.g. 3")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.boundary_weight:
        cfg.boundary_weight = args.boundary_weight
        args.settings = ", ".join(x for x in [f"boundary-weighted loss (x{args.boundary_weight:g})", args.settings] if x)
    if args.seed is not None:
        assert PATHS.splits.exists(), "--seed changes training only: the frozen splits must already exist"
        cfg.seed = args.seed
        args.settings = ", ".join(x for x in [f"training seed {args.seed}", args.settings] if x)
    if args.aug_features:
        args.settings = ", ".join(x for x in [f"stain augmentation (p={args.aug_p:g})", args.settings] if x)
    if args.class_weights:
        cfg.class_weights = True
        args.settings = ", ".join(x for x in ["class-weighted CE", args.settings] if x)

    if not PATHS.splits.exists():
        # First the held-out test set, then the folds on what is left. Both are frozen
        # to disk so every run uses the same slides.
        full = make_test_split(pd.read_csv(PATHS.train_csv), cfg.test_frac, cfg.seed)
        full[full.is_test].drop(columns="is_test").to_csv(PATHS.test_split, index=False)
        dev = make_folds(full[~full.is_test].drop(columns="is_test"), cfg.n_folds, cfg.seed)
        dev.to_csv(PATHS.splits, index=False)
    df = pd.read_csv(PATHS.splits)
    test_ids = set(pd.read_csv(PATHS.test_split).image_id)
    assert not test_ids & set(df.image_id), "test slides found in the development split"
    # the feature size depends on the encoder (1024 ResNet-50, 768 Hibou-B, 3072 Midnight): read it from the files
    cfg.feat_dim = int(torch.load(next(PATHS.features.glob("*.pt")), weights_only=True)["features"].shape[1])
    print(f"feature size per tile: {cfg.feat_dim}")

    run_dir = PATHS.runs / f"{datetime.now():%Y%m%d-%H%M}_{args.model}"
    scores = []
    for fold in args.folds if args.folds is not None else range(cfg.n_folds):
        train_ds = BagDataset(df[df.fold != fold], PATHS.features, max_tiles=args.max_tiles,
                              aug_dir=args.aug_features, aug_p=args.aug_p)
        val_ds = BagDataset(df[df.fold == fold], PATHS.features)
        print(f"\n=== fold {fold}: {len(train_ds)} train / {len(val_ds)} val slides")
        if args.seed is not None:
            torch.manual_seed(cfg.seed + fold)    # new start weights too (default runs keep their old behaviour)
        model = MODELS[args.model](feat_dim=cfg.feat_dim, hidden_dim=cfg.hidden_dim,
                                   n_classes=cfg.n_classes, dropout=cfg.dropout)
        scores.append(train_fold(model, train_ds, val_ds, cfg, run_dir / f"fold{fold}", device=args.device))

    print(f"\n{args.model}: val QWK per fold {np.round(scores, 3).tolist()}  mean {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    print(f"saved to {run_dir}")

    if args.folds is None and not args.no_log:
        log_run(run_dir, args.note, args.settings)


if __name__ == "__main__":
    main()
