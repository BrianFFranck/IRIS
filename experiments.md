# Iris: experiment log

One row per run. Add new runs at the **top** of each table. Write down what you changed and why, even when it doesn't work, because the failures belong in the write-up too.

- **val QWK** is the mean ± std over the 5 folds, taking each fold's best epoch. It's good for comparing models; the final number comes from the held-out test set, used once.
- **Run folder** is the folder under `runs/` (kept local, not on GitHub).

## Models

| Date | Model | Slides | Level | Settings | val QWK (5-fold) | GG2 called GG2 | Run folder | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | meanmax (baseline) | 299 | 1 (~5×) | 30 epochs, CE loss, lr 2e-4 | **0.47 ± 0.11** (folds 0.47 / 0.39 / 0.55 / 0.62 / 0.32) | 0 / 38 | `20261110-1707_meanmax` | Never predicts GG2; most GG2 called ISUP 0 (13) or 1 (15). Train QWK ≈ val QWK, so the model is underfitting rather than overfitting. **Number to beat.** |
| 2026-11-10 | meanmax (baseline) | 20 | 1 | 5 epochs | 0.00 | – | – | Pipeline smoke test only. Validation folds of 1–9 slides are too small to mean anything. |

## Data and tiling

| Date | What | Result | Notes |
|---|---|---|---|
| 2026-11-10 | Tiling check against PANDA label masks (`05_check_tiling.py`), 296 slides | Cancer recall, median: Karolinska **0.94**, Radboud **0.95** (10th percentile 0.89 for both). Tissue recall 0.92 / 0.87. Tile precision 0.59 / 0.73. | Worst slide keeps 78% of its cancer. Missed cancer is mostly at core edges. Idea for later: overlapping or shifted tile grid. |
| 2026-11-10 | Tissue mask: ignore pen ink, cap Otsu threshold; `min_tissue` 0.5 → 0.25 | Tiles per slide (20 slides): min 2 → 17, median 27 → 45 | Pen ink made Otsu discard pale tissue: slide `1364c10e` (ISUP 5) went from 2 tiles to 18. |
| 2026-11-10 | Downloaded 299 slides + 296 label masks | 9.8 GB + 1.1 GB | 3 slides have no mask on Kaggle. Stratified by lab and grade, seed 42. |

## Ideas to try

- [ ] Gated attention MIL (`abmil.py`): does it start recognising GG2?
- [ ] Boundary-weighted loss (GG1 ↔ GG2 errors cost more)
- [ ] Ordinal or regression loss instead of cross-entropy
- [ ] Level 0 (20×) tiles on Kaggle
- [ ] Pathology foundation model as the tile encoder instead of ImageNet ResNet-50
- [ ] Overlapping or shifted tile grid (catch cancer at core edges)
