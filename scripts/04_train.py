"""Train a MIL model on the saved feature bags with cross-validation.

Usage:
  python scripts/04_train.py --model meanmax            # the baseline
  python scripts/04_train.py --model abmil              # Iris v1 (after you finish abmil.py)
  python scripts/04_train.py --model meanmax --folds 0  # just one fold, quick

A full run (all folds) is written to experiments.md by itself. If it is a new best it is
also pushed and shown on the GitHub profile card; otherwise it stays local. Quick runs
with --folds are never logged. Use --no-log to skip logging altogether.
  python scripts/04_train.py --model abmil --note "what you learned from this run"
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from iris.config import PATHS, TrainConfig
from iris.data.dataset import BagDataset
from iris.data.splits import make_folds
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
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs

    if PATHS.splits.exists():
        df = pd.read_csv(PATHS.splits)
    else:
        df = make_folds(pd.read_csv(PATHS.train_csv), cfg.n_folds, cfg.seed)
        df.to_csv(PATHS.splits, index=False)       # freeze splits so every run uses the same

    run_dir = PATHS.runs / f"{datetime.now():%Y%m%d-%H%M}_{args.model}"
    scores = []
    for fold in args.folds if args.folds is not None else range(cfg.n_folds):
        train_ds = BagDataset(df[df.fold != fold], PATHS.features, max_tiles=args.max_tiles)
        val_ds = BagDataset(df[df.fold == fold], PATHS.features)
        print(f"\n=== fold {fold}: {len(train_ds)} train / {len(val_ds)} val slides")
        model = MODELS[args.model](feat_dim=cfg.feat_dim, hidden_dim=cfg.hidden_dim,
                                   n_classes=cfg.n_classes, dropout=cfg.dropout)
        scores.append(train_fold(model, train_ds, val_ds, cfg, run_dir / f"fold{fold}"))

    print(f"\n{args.model}: val QWK per fold {np.round(scores, 3).tolist()}  mean {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    print(f"saved to {run_dir}")

    if args.folds is None and not args.no_log:
        log_run(run_dir, args.note, args.settings)


if __name__ == "__main__":
    main()
