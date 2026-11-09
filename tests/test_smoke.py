"""Smoke tests on FAKE data: no slides, no internet needed. Run: pytest tests/

They check shapes and that the pieces fit together, not that Iris is any good.
`test_abmil_*` fail until you finish iris/models/abmil.py, which is how you
know when you're done.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from iris.config import TrainConfig
from iris.data.dataset import BagDataset
from iris.data.splits import make_folds
from iris.data.tissue import tissue_mask
from iris.metrics import qwk
from iris.models import GatedAttentionMIL, MeanMaxPoolMIL
from iris.train import train_fold

D = 1024


def test_tissue_mask_finds_pink_blob():
    img = np.full((200, 200, 3), 240, dtype=np.uint8)       # light grey glass
    img[50:150, 50:150] = (200, 100, 170)                    # pink "tissue"
    m = tissue_mask(img)
    assert m[100, 100] and not m[10, 10]


def test_qwk_perfect_and_bad():
    assert qwk([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert qwk([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]) < 0


def test_folds_cover_everything():
    df = pd.DataFrame({"image_id": [f"s{i}" for i in range(60)],
                       "data_provider": ["karolinska", "radboud"] * 30,
                       "isup_grade": [i % 6 for i in range(60)]})
    f = make_folds(df, n_folds=5)
    assert set(f.fold) == set(range(5)) and (f.fold >= 0).all()


def test_baseline_shapes():
    out = MeanMaxPoolMIL()(torch.randn(37, D))
    assert out["logits"].shape == (1, 6)


def test_abmil_shapes():
    model = GatedAttentionMIL().eval()
    out = model(torch.randn(37, D))
    assert out["logits"].shape == (1, 6)
    assert out["attention"].shape == (37,)
    assert out["attention"].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (out["attention"] >= 0).all()


def test_abmil_handles_one_tile():
    out = GatedAttentionMIL().eval()(torch.randn(1, D))
    assert out["attention"].item() == pytest.approx(1.0)


def test_training_end_to_end(tmp_path):
    """Fake bags where the label is hidden in the features: loss must go down."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        grade = i % 6
        feats = torch.randn(rng.integers(5, 40), D)
        feats[:, :6] *= 0.1
        feats[:, grade] += 3.0                               # a signal the model can find
        torch.save({"features": feats.half(), "coords": torch.zeros(len(feats), 2)}, tmp_path / f"s{i}.pt")
        rows.append({"image_id": f"s{i}", "isup_grade": grade})
    df = pd.DataFrame(rows)
    ds = BagDataset(df, tmp_path)
    cfg = TrainConfig(epochs=8, lr=1e-3)
    best = train_fold(MeanMaxPoolMIL(), ds, ds, cfg, tmp_path / "run")
    assert best > 0.8
