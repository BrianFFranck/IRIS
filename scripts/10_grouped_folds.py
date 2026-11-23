"""Leakage check, step B: new folds that keep very similar slides together.

PANDA has no patient IDs, so similarity stands in for "probably the same patient". Two slides of
the same centre whose summary vectors (see scripts/09_leakage_check.py) have a cosine similarity
at or above that centre's close-twin cut are linked; linked slides, and everything linked to them
in a chain, form one group, and a group never gets split across folds. The cuts are the ones
step A used to define a close twin (top 10% of twin similarity): Radboud 0.960, Karolinska 0.914.

The folds are then made like the original ones (5 folds, stratified by centre and grade, seed 42),
but with the groups kept whole. Only development slides are used: the test set stays locked.
The slide without features keeps its own one-slide group.

Output: kaggle/splits/splits_grouped.csv (same columns as splits.csv, plus `group`).

Usage:
  python scripts/10_grouped_folds.py runs_kaggle/slide_embeddings/slide_embeddings.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from iris.config import PATHS

CUTS = {"radboud": 0.960, "karolinska": 0.914}     # close-twin cuts from step A (2026-11-23)
OUT = Path(__file__).resolve().parents[1] / "kaggle" / "splits" / "splits_grouped.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings", type=Path)
    args = ap.parse_args()

    emb = np.load(args.embeddings)
    row_of = pd.Series(range(len(emb["image_id"])), index=emb["image_id"])
    df = pd.read_csv(PATHS.splits)            # development slides only
    df["group"] = ""

    for centre, cut in CUTS.items():
        ids = df[(df.data_provider == centre) & df.image_id.isin(row_of.index)].image_id
        m = emb["mean"][row_of[ids].values].astype(np.float64)
        x = (m - m.mean(0)) / (m.std(0) + 1e-8)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        sim = x @ x.T
        np.fill_diagonal(sim, 0)
        n, label = connected_components(csr_matrix(sim >= cut), directed=False)
        df.loc[ids.index, "group"] = [f"{centre}_{g}" for g in label]
        sizes = np.bincount(label)
        print(f"{centre}: {len(ids)} slides -> {n} groups; {sizes[sizes > 1].sum()} slides in "
              f"{(sizes > 1).sum()} groups of 2 or more; largest {sizes.max()}")

    lonely = df.group == ""
    df.loc[lonely, "group"] = "single_" + df.loc[lonely, "image_id"]

    strat = df.data_provider + "_" + df.isup_grade.astype(str)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, strat, groups=df.group)):
        df.loc[df.index[val_idx], "fold"] = fold

    assert (df.groupby("group").fold.nunique() == 1).all()
    print("\nslides per fold and centre:")
    print(pd.crosstab(df.fold, df.data_provider))
    print("\nshare of each ISUP grade per fold (%):")
    print((pd.crosstab(df.fold, df.isup_grade, normalize="index") * 100).round(1))
    df.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
