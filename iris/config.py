"""One place for paths and knobs."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Override with env vars when running elsewhere, e.g. on Kaggle:
#   IRIS_PANDA_DIR=/kaggle/input/prostate-cancer-grade-assessment IRIS_DATA_DIR=/kaggle/working
DATA = Path(os.environ.get("IRIS_DATA_DIR", ROOT / "data"))


@dataclass
class Paths:
    data: Path = DATA
    panda: Path = Path(os.environ.get("IRIS_PANDA_DIR", DATA / "panda"))  # train.csv + train_images/
    coords: Path = DATA / "coords"        # one .npy of tile coordinates per slide
    features: Path = DATA / "features"    # one .pt feature bag per slide
    splits: Path = DATA / "splits.csv"
    runs: Path = DATA / "runs" if "IRIS_DATA_DIR" in os.environ else ROOT / "runs"                     # checkpoints, logs, predictions

    @property
    def train_csv(self) -> Path:
        return self.panda / "train.csv"

    @property
    def images(self) -> Path:
        return self.panda / "train_images"

    @property
    def masks(self) -> Path:
        return self.panda / "train_label_masks"


@dataclass
class TileConfig:
    # PANDA .tiff files have 3 levels: 0 (full res, ~0.5 um/px), 1 (4x down), 2 (16x down).
    level: int = 1
    tile_size: int = 256          # pixels, at `level`
    mask_level: int = -1          # -1 = lowest-resolution level, used only for the tissue mask
    min_tissue: float = 0.25      # keep a tile if at least this fraction is tissue (0.5 missed thin cores)


@dataclass
class TrainConfig:
    n_classes: int = 6            # ISUP 0..5
    feat_dim: int = 1024          # ResNet-50 truncated after layer3
    hidden_dim: int = 256
    dropout: float = 0.25
    lr: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    n_folds: int = 5
    seed: int = 42


PATHS = Paths()
