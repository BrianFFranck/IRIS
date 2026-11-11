"""Train/validation splits.

PANDA comes from two labs: Karolinska (Sweden) and Radboud (Netherlands).
They stain and scan differently, and they labelled differently. If you split
carelessly the validation score lies to you. So we stratify on BOTH the ISUP
grade and the data provider.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Slides removed before any split (see experiments.md, 2026-11-11).
EXCLUDE = {"b0a92a74cb53899311acc30b7405e101"}   # labelled 4+3 but ISUP 2


def make_test_split(df: pd.DataFrame, test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Add an `is_test` column. True = held-out test slide, used once at the very end."""
    df = df[~df["image_id"].isin(EXCLUDE)].copy()
    strat = df["data_provider"].astype(str) + "_" + df["isup_grade"].astype(str)
    dev_idx, test_idx = train_test_split(
        df.index, test_size=test_frac, stratify=strat,random_state=seed)
    df["is_test"] = df.index.isin(test_idx)
    return df


def make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a `fold` column (0..n_folds-1) to the PANDA train.csv dataframe."""
    df = df.copy()
    strat = df["data_provider"].astype(str) + "_" + df["isup_grade"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, strat)):
        df.loc[df.index[val_idx], "fold"] = fold
    return df
