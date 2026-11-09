# Iris

**An AI that grades prostate biopsies and shows where it looked.**

Iris predicts the ISUP Grade Group of a prostate biopsy from its whole-slide image (WSI). It also produces an attention heatmap of the tissue regions that drove the prediction. It focuses on the boundary between **Grade Group 1 (Gleason 3+3)** and **Grade Group 2 (Gleason 3+4)**. There, pathologists often disagree, and the grade can decide between active surveillance and treatment.

> **Status: early R&D.** The pipeline code is written and tested on synthetic data. It has not yet been run on real slides, and there are no results yet.
> Iris is a research prototype. It is **not a medical device** and must not be used for clinical decisions.

---

## How it works

A WSI can be around 100,000 × 100,000 pixels and carries a single label, so Iris works in four stages:

```
slide.tiff
   │ 1. tissue mask (Otsu on saturation) → grid of 256×256 tissue tiles
   ▼
tiles
   │ 2. frozen ResNet-50 (truncated after layer 3) → 1024-d vector per tile, saved once
   ▼
feature bag (N tiles × 1024)
   │ 3. attention-based multiple instance learning → one grade per slide
   ▼
ISUP grade + attention weights
   │ 4. attention painted back onto the slide
   ▼
heatmap a pathologist can check
```

Stage 2 is the expensive one and runs once. Every experiment after it trains on the saved features in minutes.

## Approach

| Model | Role |
| --- | --- |
| Mean/max pooling | Baseline, no attention |
| Gated attention MIL ([Ilse et al. 2018](https://arxiv.org/abs/1802.04712); the core of [CLAM](https://github.com/mahmoodlab/CLAM)) | Iris v1 |
| Attention MIL + boundary-weighted loss | Penalises GG1 ↔ GG2 errors more heavily |

The evaluation uses quadratic weighted kappa (QWK) overall and per data centre, plus sensitivity and specificity at the GG1 vs GG2 boundary. Folds are stratified by grade and centre, and a held-out test set is used once at the end.

## Roadmap

- [x] Pipeline skeleton: tissue masking, tiling, feature extraction, training loop, baseline
- [ ] Data: explore PANDA, tile and extract features on a development subset
- [ ] Baseline: mean/max pooling results with cross-validation
- [ ] Iris v1: gated attention MIL
- [ ] Boundary-weighted loss and ablations
- [ ] Full PANDA run at 20× on GPU
- [ ] Evaluation: QWK, per-centre results, GG1 vs GG2 analysis
- [ ] Explainability: attention heatmaps, checked against Radboud's pixel-level Gleason masks
- [ ] Write-up

## Project layout

```
iris/
  config.py            paths and settings (env vars IRIS_PANDA_DIR / IRIS_DATA_DIR override paths)
  data/tissue.py       tissue mask
  data/tiling.py       tile grid (stores coordinates, not images)
  data/splits.py       cross-validation folds stratified by grade and centre
  data/dataset.py      one feature bag per slide
  features.py          ResNet-50 feature extractor
  models/pooling.py    mean/max baseline
  models/abmil.py      gated attention MIL (in progress)
  train.py             training loop for one fold
  metrics.py           QWK and boundary analysis
  heatmap.py           attention heatmaps (planned)
scripts/
  01_download_subset.py   download a stratified subset of PANDA from Kaggle
  02_tile_slides.py       tissue mask + tile coordinates for every slide
  03_extract_features.py  one feature bag per slide
  04_train.py             cross-validated training
tests/
  test_smoke.py        shape and end-to-end checks on synthetic data
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). OpenSlide is installed from pip (`openslide-bin`), so no system package is needed.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish

# CPU-only PyTorch (drop the index flags on a machine with an NVIDIA GPU)
uv pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match -e ".[dev]"

pytest tests/
```

## Running

You need a Kaggle account, the PANDA competition rules accepted, and an API token in `~/.kaggle/kaggle.json`.

```bash
python scripts/01_download_subset.py --n 300   # ~300 slides for local development
python scripts/02_tile_slides.py
python scripts/03_extract_features.py          # add --device cuda on a GPU
python scripts/04_train.py --model meanmax
```

To run on all of PANDA in a Kaggle notebook (the data is already mounted there):

```bash
git clone https://github.com/BrianFFranck/iris.git && cd iris && pip install -e .
export IRIS_PANDA_DIR=/kaggle/input/prostate-cancer-grade-assessment
export IRIS_DATA_DIR=/kaggle/working
python scripts/02_tile_slides.py && python scripts/03_extract_features.py --device cuda
```

## Data

Iris is developed on the public [PANDA challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment) dataset: 10,616 biopsy slides from Radboud University Medical Center and Karolinska Institutet ([Bulten et al. 2022, *Nature Medicine*](https://www.nature.com/articles/s41591-021-01620-2)). The data is not included in this repository. It must be downloaded under the competition's terms, which restrict it to non-commercial use.

## License

Copyright © 2026 Brian Franck. All rights reserved. The source is visible for reference and review. See [LICENSE](LICENSE).

## Author

**Brian Franck**, BSc Computational Biomedicine & AI, Universidade Lusófona, Lisbon. GitHub: [@BrianFFranck](https://github.com/BrianFFranck)
