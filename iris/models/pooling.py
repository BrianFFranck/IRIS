"""BASELINE: the dumbest reasonable MIL model.

Average and max the tile features, concatenate, classify. No attention, no
learning of "which tiles matter". Iris v1 has to beat this, otherwise the
attention is not earning its place. Finished; use it as a reference.
"""
import torch
import torch.nn as nn


class MeanMaxPoolMIL(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256, n_classes=6, dropout=0.25):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, h: torch.Tensor) -> dict:
        # h: (N, D) one bag. Returns logits (1, n_classes)
        z = torch.cat([h.mean(0), h.max(0).values], dim=-1)
        return {"logits": self.head(z).unsqueeze(0), "attention": None}
