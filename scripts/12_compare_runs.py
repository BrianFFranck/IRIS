"""Compare 5-fold runs on the same slides: QWK AND the GG1 vs GG2 boundary, with 95% CIs.

Lesson of 2026-11-25: QWK can stay the same while GG2 detection falls (stain normalisation,
Radboud 374 -> 345 GG2 slides caught). So every comparison should also look at the boundary
this project is about. For each run, per centre and for all slides:
  - QWK, with a bootstrap 95% CI
  - GG2 caught: GG2 slides called GG2 (sensitivity for GG2)
  - GG1 called GG2: GG1 slides wrongly called GG2 (over-grading at the boundary)
  - GG2 called GG1: GG2 slides wrongly called GG1 (pattern 4 missed)
Then every run against the FIRST one (the reference), on the same slides:
  - change in QWK, with a paired bootstrap 95% CI
  - change in GG2 caught and in GG1 called GG2, with a McNemar test (exact: only the slides
    where the two runs disagree count)

Only validation predictions of the development slides are used: the test set stays locked.

Usage:
  python scripts/12_compare_runs.py runs_kaggle/trainhibou/runs/20261122-1942_abmil \
                                    runs_kaggle/trainhiboustain/runs/20261125-1648_abmil
  A run trained with other folds needs its splits file:
    --splits runs_kaggle/traingrouped/runs/20261123-1941_abmil kaggle/splits/splits_grouped.csv
  --names "original colours" "stain"      short names for the table (default: folder names)
  --out FILE.json                         also save every number
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from iris.config import PATHS
from iris.metrics import qwk

# The one development slide without tissue tiles, so without features: no run predicts it.
NO_FEATURES = {"3790f55cad63053e956fb73027179707"}
N_BOOT = 2000


def fast_qwk(yt, yp, k=6):
    """Same number as iris.metrics.qwk, but fast enough for thousands of bootstrap samples."""
    o = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    return 1 - (w * o).sum() / (w * e).sum()


def load_run(run_dir, splits_file):
    """Validation predictions of all 5 folds, as one table indexed by image_id."""
    df = pd.read_csv(splits_file)
    parts = []
    for k in sorted(df.fold.unique()):
        p = json.load(open(Path(run_dir) / f"fold{k}" / "val_preds.json"))
        fold = df[df.fold == k]
        if len(fold) != len(p["y_true"]):
            fold = fold[~fold.image_id.isin(NO_FEATURES)]
        assert list(fold.isup_grade) == p["y_true"], f"{run_dir} fold {k}: slides out of order"
        parts.append(pd.DataFrame({"image_id": fold.image_id.values, "centre": fold.data_provider.values,
                                   "y": fold.isup_grade.values, "pred": p["y_pred"]}))
    return pd.concat(parts).set_index("image_id")


def boundary(y, p):
    gg1, gg2 = y == 1, y == 2
    return {"gg2_caught": int((p[gg2] == 2).sum()), "n_gg2": int(gg2.sum()),
            "gg1_called_gg2": int((p[gg1] == 2).sum()), "n_gg1": int(gg1.sum()),
            "gg2_called_gg1": int((p[gg2] == 1).sum())}


def mcnemar(a_hit, b_hit):
    """Exact McNemar: of the slides where the two runs disagree, does one side win more often?"""
    only_a, only_b = int((a_hit & ~b_hit).sum()), int((b_hit & ~a_hit).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help="run folders; the first one is the reference")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--splits", nargs=2, action="append", default=[], metavar=("RUN", "SPLITS"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    names = args.names or [r.name for r in args.runs]
    assert len(names) == len(args.runs)
    splits = {Path(r).resolve(): Path(s) for r, s in args.splits}

    runs = [load_run(r, splits.get(r.resolve(), PATHS.splits)) for r in args.runs]
    ids = runs[0].index
    for r, n in zip(runs[1:], names[1:]):
        assert set(r.index) == set(ids), f"{n} predicts other slides than {names[0]}"
    runs = [r.loc[ids] for r in runs]
    y, centre = runs[0].y.values, runs[0].centre.values
    groups = {"radboud": centre == "radboud", "karolinska": centre == "karolinska", "all": np.ones(len(y), bool)}

    rng = np.random.default_rng(42)
    boot = {g: [np.flatnonzero(m)[rng.integers(0, m.sum(), m.sum())] for _ in range(N_BOOT)] for g, m in groups.items()}
    assert abs(fast_qwk(y, runs[0].pred.values) - qwk(y, runs[0].pred.values)) < 1e-9

    out = {"runs": dict(zip(names, map(str, args.runs))), "slides": len(y), "groups": {}}
    for g, m in groups.items():
        print(f"\n=== {g} ({m.sum()} slides) ===")
        res = {}
        for n, r in zip(names, runs):
            p = r.pred.values
            q = [fast_qwk(y[i], p[i]) for i in boot[g]]
            ci_q = np.percentile(q, [2.5, 97.5])
            b = boundary(y[m], p[m])
            gg2 = np.flatnonzero(m & (y == 2))
            gg1 = np.flatnonzero(m & (y == 1))
            sens = [(p[i] == 2).mean() for i in (gg2[rng.integers(0, len(gg2), len(gg2))] for _ in range(N_BOOT))]
            fp = [(p[i] == 2).mean() for i in (gg1[rng.integers(0, len(gg1), len(gg1))] for _ in range(N_BOOT))]
            res[n] = {"qwk": float(fast_qwk(y[m], p[m])), "qwk_ci95": ci_q.tolist(), **b,
                      "gg2_caught_ci95": np.percentile(sens, [2.5, 97.5]).tolist(),
                      "gg1_called_gg2_ci95": np.percentile(fp, [2.5, 97.5]).tolist()}
            s, f = np.percentile(sens, [2.5, 97.5]), np.percentile(fp, [2.5, 97.5])
            print(f"{n:>20}: QWK {res[n]['qwk']:.3f} ({ci_q[0]:.3f}-{ci_q[1]:.3f}) | "
                  f"GG2 caught {b['gg2_caught']}/{b['n_gg2']} = {b['gg2_caught'] / b['n_gg2']:.1%} ({s[0]:.1%}-{s[1]:.1%}) | "
                  f"GG1 called GG2 {b['gg1_called_gg2']}/{b['n_gg1']} = {b['gg1_called_gg2'] / b['n_gg1']:.1%} ({f[0]:.1%}-{f[1]:.1%}) | "
                  f"GG2 called GG1 {b['gg2_called_gg1']}")
        ref = runs[0].pred.values
        for n, r in zip(names[1:], runs[1:]):
            p = r.pred.values
            d = [fast_qwk(y[i], p[i]) - fast_qwk(y[i], ref[i]) for i in boot[g]]
            ci = np.percentile(d, [2.5, 97.5])
            gg2, gg1 = m & (y == 2), m & (y == 1)
            lost, won, p2 = mcnemar(ref[gg2] == 2, p[gg2] == 2)
            fa, fb, p1 = mcnemar(ref[gg1] == 2, p[gg1] == 2)
            res[n]["vs_reference"] = {"qwk_change": float(fast_qwk(y[m], p[m]) - fast_qwk(y[m], ref[m])), "qwk_change_ci95": ci.tolist(),
                                      "gg2_caught_only_reference": lost, "gg2_caught_only_this": won, "gg2_mcnemar_p": p2,
                                      "gg1_called_gg2_only_reference": fa, "gg1_called_gg2_only_this": fb, "gg1_mcnemar_p": p1}
            v = res[n]["vs_reference"]
            print(f"{'':>20}  vs {names[0]}: QWK {v['qwk_change']:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f}) | "
                  f"GG2 caught {won - lost:+d} (only ref {lost}, only this {won}, p = {p2:.3f}) | "
                  f"GG1 called GG2 {fb - fa:+d} (p = {p1:.3f})  [{n}]")
        out["groups"][g] = res

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
