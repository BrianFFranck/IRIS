"""PyTorch dataset of feature bags: one item = one slide.

Each slide becomes a tensor of shape (N_tiles, feat_dim), with ONE label
(the ISUP grade). N_tiles is different for every slide, which is why MIL
models are trained with batch_size=1.
"""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features_dir: Path, max_tiles: int | None = None):
        # drop slides whose features were not extracted (yet)
        have = {p.stem for p in Path(features_dir).glob("*.pt")}
        self.df = df[df["image_id"].isin(have)].reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.max_tiles = max_tiles

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        bag = torch.load(self.features_dir / f"{row.image_id}.pt", weights_only=True)
        feats = bag["features"].float()          # (N, D)
        if self.max_tiles and len(feats) > self.max_tiles:
            # random subset during training = cheap augmentation + bounded memory
            idx = torch.randperm(len(feats))[: self.max_tiles]
            feats = feats[idx]
        label = torch.tensor(int(row.isup_grade), dtype=torch.long)
        return feats, label, row.image_id
