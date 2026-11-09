"""Train/validation splits.

PANDA comes from two labs: Karolinska (Sweden) and Radboud (Netherlands).
They stain and scan differently, and they labelled differently. If you split
carelessly the validation score lies to you. So we stratify on BOTH the ISUP
grade and the data provider.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a `fold` column (0..n_folds-1) to the PANDA train.csv dataframe."""
    df = df.copy()
    strat = df["data_provider"].astype(str) + "_" + df["isup_grade"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, strat)):
        df.loc[df.index[val_idx], "fold"] = fold
    return df

# TODO (before reporting final numbers): keep a held-out TEST set that you never
# touch during development (e.g. 10-15% of slides, stratified the same way), and
# run cross-validation only on the rest. Decide this early and write it down.
