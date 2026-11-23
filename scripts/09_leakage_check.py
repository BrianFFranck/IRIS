"""Leakage check, step A: do validation slides with a very similar slide in training get graded
right more often?

PANDA has several biopsies per patient but no patient IDs, so the 5 folds (made per slide) may put
two biopsies of the same patient on both sides. If the model partly "recognises the patient",
validation slides with a close twin in the training folds would be graded right more often.

1. Every slide becomes one vector: the average of its tile features (kaggle/slide_embeddings).
   Each centre's vectors are rescaled to mean 0 and spread 1 per number, so colour differences
   between the hospitals do not dominate.
2. Twin of a slide = the most similar slide (cosine similarity) of the SAME centre in the OTHER
   folds, i.e. among the slides the model trained on when this slide was validated. Patients
   never cross hospitals, so a twin from the other centre can never be the same patient: the
   twin similarity to the other centre is the reference for "not the same patient".
3. Close twin = twin similarity in the top 10% of its centre. Accuracy (exact ISUP grade) of the
   validation predictions is compared close vs rest WITHIN each grade, then averaged with the
   close group's grade mix as weights, so a different grade mix cannot fake a difference.

Weakness: typical-looking slides have closer twins and may also be easier, so a gap is a warning,
not proof. The strong test is step B: folds that keep similar slides together, then retrain.

Usage:
  python scripts/09_leakage_check.py EMBEDDINGS.npz RUN_DIR
  e.g. runs_kaggle/slide_embeddings/slide_embeddings.npz runs_kaggle/trainhibou/runs/20261122-1942_abmil
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iris.config import PATHS

CENTRES = ["radboud", "karolinska"]
CLOSE_TOP = 0.10


def load_val_preds(df, run_dir):
    """Put each fold's validation predictions next to its slides (same order as BagDataset)."""
    df = df.copy()
    df["pred"] = -1
    for k in sorted(df.fold.unique()):
        p = json.load(open(Path(run_dir) / f"fold{k}" / "val_preds.json"))
        idx = df.index[df.fold == k]
        assert list(df.loc[idx, "isup_grade"]) == p["y_true"], f"fold {k}: slides out of order"
        df.loc[idx, "pred"] = p["y_pred"]
    return df


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def grade_adjusted_gap(sub):
    """Accuracy close minus rest, compared within each grade, weighted by the close group's grades."""
    gap, weight = 0.0, 0
    for g, part in sub.groupby("isup_grade"):
        close, rest = part[part.close], part[~part.close]
        if len(close) and len(rest):
            gap += len(close) * (close.correct.mean() - rest.correct.mean())
            weight += len(close)
    return gap / weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings", type=Path)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    emb = np.load(args.embeddings)
    vec = pd.DataFrame({"image_id": emb["image_id"]}).reset_index().set_index("image_id")["index"]
    df = pd.read_csv(PATHS.splits)            # development slides only: the test set stays locked
    df = df[df.image_id.isin(vec.index)].reset_index(drop=True)
    df = load_val_preds(df, args.run_dir)
    df["correct"] = df.pred == df.isup_grade

    # rescale per centre, then unit length so that a dot product = cosine similarity
    x = {}
    for c in CENTRES:
        m = emb["mean"][vec[df[df.data_provider == c].image_id].values].astype(np.float64)
        x[c] = unit((m - m.mean(0)) / (m.std(0) + 1e-8))

    df["twin_sim"], df["other_centre_sim"] = np.nan, np.nan
    for c in CENTRES:
        rows = df.index[df.data_provider == c]
        other = [o for o in CENTRES if o != c][0]
        sim = x[c] @ x[c].T
        folds = df.loc[rows, "fold"].values
        sim[folds[:, None] == folds[None, :]] = -np.inf     # only slides in the other folds
        df.loc[rows, "twin_sim"] = sim.max(1)
        df.loc[rows, "other_centre_sim"] = (x[c] @ x[other].T).max(1)

    print(f"{len(df)} development slides\n")
    out = {}
    for c in CENTRES:
        sub = df[df.data_provider == c].copy()
        cut = sub.twin_sim.quantile(1 - CLOSE_TOP)
        sub["close"] = sub.twin_sim >= cut
        ref99 = sub.other_centre_sim.quantile(0.99)
        beyond = int((sub.twin_sim > ref99).sum())
        gap = grade_adjusted_gap(sub)
        rng = np.random.default_rng(42)
        boots = [grade_adjusted_gap(sub.iloc[rng.integers(0, len(sub), len(sub))]) for _ in range(1000)]
        ci = np.percentile(boots, [2.5, 97.5])
        out[c] = {"slides": len(sub), "twin_sim_median": float(sub.twin_sim.median()),
                  "other_centre_sim_median": float(sub.other_centre_sim.median()),
                  "twins_beyond_other_centre_p99": beyond, "close_cut": float(cut),
                  "acc_close": float(sub[sub.close].correct.mean()), "acc_rest": float(sub[~sub.close].correct.mean()),
                  "grade_adjusted_gap": float(gap), "gap_ci95": ci.tolist()}
        print(f"{c} ({len(sub)} slides)")
        print(f"  twin similarity, median: same centre {sub.twin_sim.median():.3f} | "
              f"other centre (never the same patient) {sub.other_centre_sim.median():.3f}")
        print(f"  twins closer than 99% of other-centre twins: {beyond} ({beyond / len(sub):.0%})")
        print(f"  accuracy, close twin (top 10%, sim >= {cut:.3f}): {sub[sub.close].correct.mean():.3f} | "
              f"rest: {sub[~sub.close].correct.mean():.3f}")
        print(f"  gap within the same grade: {gap:+.3f} (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})")
        for g, part in sub.groupby("isup_grade"):
            cl, re = part[part.close], part[~part.close]
            print(f"    ISUP {g}: close {cl.correct.mean():.2f} (n={len(cl)}) | rest {re.correct.mean():.2f} (n={len(re)})")
        print()

    run_dir = PATHS.runs / "leakage_check"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(run_dir / "results.json", "w"), indent=1)
    df[["image_id", "data_provider", "fold", "isup_grade", "pred", "twin_sim", "other_centre_sim"]].to_csv(
        run_dir / "twins.csv", index=False)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
