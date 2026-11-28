"""Compare Iris's grades on PAR (external slides) with the PAR pathologists, and check the PAR rule.

The rule (experiments.md, 2026-11-26), written before the run:
  Iris generalises to PAR if its QWK against EACH pathologist (1 and 2) is at least 0.81, and
  below 10% of the slides BOTH pathologists call benign are called ISUP 3 or higher.
  0.81 = how well pathologists 1 and 2 agree with each other.

For each model (a grades CSV from scripts/14_grade_external.py):
  - QWK against pathologist 1, 2 and 3 (3 graded only 59 slides), with a bootstrap 95% CI
  - the benign check: of the slides both pathologists 1 and 2 call benign, how many Iris calls
    ISUP 1-2 and how many ISUP 3 or higher (with an exact 95% CI for the share)
  - how far off: share of slides graded higher / the same / lower than pathologist 1
  - confusion table against pathologist 1
Then every model against the FIRST one: change in QWK against each pathologist, with a paired
bootstrap 95% CI (same slides, same bootstrap samples).
The human benchmark, pathologist 1 against pathologist 2, is printed on the same slides.

Usage:
  python scripts/15_external_vs_pathologists.py par_original.csv par_stain.csv par_aug.csv \
      --names original stain augmented --out runs/par/par_result.json
  --labels data/external/par/slide_labels.tsv   (the default; from the BioImage Archive)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

N_BOOT = 2000
RATERS = {"pathologist 1": "isup_1", "pathologist 2": "isup_r2", "pathologist 3": "isup_r3"}
RULE_QWK, RULE_BENIGN = 0.81, 0.10


def fast_qwk(yt, yp, k=6):
    o = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    return 1 - (w * o).sum() / (w * e).sum()


def slide_id(stem: str) -> str:
    """Feature file c001a_grundium -> label C001A."""
    return stem.split("_")[0].upper()


def load_labels(path):
    lab = pd.read_csv(path, sep="\t").set_index("slide_id")
    for c in RATERS.values():
        lab[c] = pd.to_numeric(lab[c], errors="coerce")        # empty = that pathologist did not grade it
    return lab


def load_grades(path):
    g = pd.read_csv(path)
    g.index = g.slide.map(slide_id)
    assert g.index.is_unique, f"{path}: a slide appears twice"
    return g.grade.astype(int)


def qwk_ci(y, p, idx):
    return [float(np.quantile([fast_qwk(y[i], p[i]) for i in idx], q)) for q in (0.025, 0.975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grades", nargs="+", type=Path, help="grades CSVs; the first one is the reference model")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--labels", type=Path, default=Path("data/external/par/slide_labels.tsv"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    names = args.names or [p.stem for p in args.grades]
    assert len(names) == len(args.grades)

    lab = load_labels(args.labels)
    preds = {n: load_grades(p) for n, p in zip(names, args.grades)}
    unknown = set().union(*[set(p.index) for p in preds.values()]) - set(lab.index)
    assert not unknown, f"slides not in the labels: {sorted(unknown)[:5]}"
    # all models on the same slides: those every model graded
    common = sorted(set(lab.index).intersection(*[set(p.index) for p in preds.values()]))
    print(f"{len(lab)} PAR slides with labels; graded by every model: {len(common)}")
    for n, p in preds.items():
        if len(p) != len(common):
            print(f"  {n}: {len(p)} slides graded, {len(p) - len(common)} not graded by all models (left out)")
    missing = sorted(set(lab.index) - set(common))
    if missing:
        print(f"  not graded (download or tiling failed?): {len(missing)} slides, e.g. {missing[:8]}")
    lab = lab.loc[common]
    rng = np.random.default_rng(0)
    out = {"n_slides": len(common), "not_graded": missing, "models": {}, "humans": {}, "changes": {}}

    # the human benchmark, on the slides both graded
    both = lab[["isup_1", "isup_r2"]].dropna().astype(int)
    y1, y2 = both.isup_1.values, both.isup_r2.values
    idx = [rng.integers(0, len(y1), len(y1)) for _ in range(N_BOOT)]
    h = {"qwk": float(fast_qwk(y1, y2)), "ci": qwk_ci(y1, y2, idx), "n": len(y1),
         "same_grade": float((y1 == y2).mean())}
    out["humans"]["pathologist 1 vs 2"] = h
    print(f"\nHuman benchmark, pathologist 1 vs 2 ({h['n']} slides): QWK {h['qwk']:.3f} "
          f"[{h['ci'][0]:.3f}, {h['ci'][1]:.3f}], same grade on {h['same_grade']:.0%}")

    benign = lab[(lab.isup_1 == 0) & (lab.isup_r2 == 0)].index
    boot = {}                                  # rater -> fixed bootstrap samples, shared by all models
    for n, p in preds.items():
        p = p.loc[common]
        m = {"qwk": {}}
        print(f"\n=== {n} ===")
        for rater, col in RATERS.items():
            ok = lab[col].notna().values
            y, yp = lab[col].values[ok].astype(int), p.values[ok]
            if len(y) < 10:
                continue
            if rater not in boot:
                boot[rater] = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
            q = {"qwk": float(fast_qwk(y, yp)), "ci": qwk_ci(y, yp, boot[rater]), "n": int(len(y)),
                 "same_grade": float((y == yp).mean()),
                 "higher": float((yp > y).mean()), "lower": float((yp < y).mean())}
            m["qwk"][rater] = q
            print(f"  vs {rater} ({q['n']} slides): QWK {q['qwk']:.3f} [{q['ci'][0]:.3f}, {q['ci'][1]:.3f}] | "
                  f"same grade {q['same_grade']:.0%}, Iris higher {q['higher']:.0%}, lower {q['lower']:.0%}")
        pb = p.loc[benign]
        k_hi = int((pb >= 3).sum())
        ci = binomtest(k_hi, len(pb)).proportion_ci() if len(pb) else None
        m["benign"] = {"n": int(len(pb)), "called_benign": int((pb == 0).sum()),
                       "called_isup_1_2": int(pb.isin([1, 2]).sum()), "called_isup_3_plus": k_hi,
                       "share_3_plus": k_hi / len(pb) if len(pb) else None,
                       "ci": [ci.low, ci.high] if ci else None}
        b = m["benign"]
        print(f"  benign by both pathologists ({b['n']} slides): Iris says benign {b['called_benign']}, "
              f"ISUP 1-2 {b['called_isup_1_2']}, ISUP 3+ {k_hi} = {b['share_3_plus']:.0%} "
              f"[{b['ci'][0]:.0%}, {b['ci'][1]:.0%}]")
        y = lab.isup_1.values.astype(int)
        conf = pd.crosstab(pd.Series(y, name="pathologist 1"), pd.Series(p.values, name="Iris"),
                           dropna=False).reindex(index=range(6), columns=range(6), fill_value=0)
        m["confusion_vs_pathologist_1"] = conf.values.tolist()
        print("  confusion table (rows: pathologist 1, columns: Iris)")
        print("    " + conf.to_string().replace("\n", "\n    "))
        q1, q2 = m["qwk"]["pathologist 1"]["qwk"], m["qwk"]["pathologist 2"]["qwk"]
        m["rule"] = {"qwk_pass": bool(q1 >= RULE_QWK and q2 >= RULE_QWK),
                     "benign_pass": bool(b["share_3_plus"] is not None and b["share_3_plus"] < RULE_BENIGN)}
        m["rule"]["pass"] = m["rule"]["qwk_pass"] and m["rule"]["benign_pass"]
        print(f"  RULE: QWK >= {RULE_QWK} vs both: {'yes' if m['rule']['qwk_pass'] else 'NO'} "
              f"({q1:.3f}, {q2:.3f}) | benign called ISUP 3+ below {RULE_BENIGN:.0%}: "
              f"{'yes' if m['rule']['benign_pass'] else 'NO'} ({b['share_3_plus']:.0%}) "
              f"-> {'PASS' if m['rule']['pass'] else 'FAIL'}")
        out["models"][n] = m

    ref = names[0]
    for n in names[1:]:
        print(f"\n=== {n} minus {ref} (same slides, paired) ===")
        out["changes"][n] = {}
        for rater, col in RATERS.items():
            if rater not in boot:
                continue
            ok = lab[col].notna().values
            y = lab[col].values[ok].astype(int)
            a, c = preds[ref].loc[common].values[ok], preds[n].loc[common].values[ok]
            d = fast_qwk(y, c) - fast_qwk(y, a)
            ds = [fast_qwk(y[i], c[i]) - fast_qwk(y[i], a[i]) for i in boot[rater]]
            ci = [float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))]
            out["changes"][n][rater] = {"qwk_change": float(d), "ci": ci}
            print(f"  QWK change vs {rater}: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        bn, br = out["models"][n]["benign"], out["models"][ref]["benign"]
        print(f"  benign slides called ISUP 3+: {br['called_isup_3_plus']} -> {bn['called_isup_3_plus']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
