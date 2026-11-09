"""IRIS v1: gated attention MIL (the core of CLAM-SB).

    h'_k = ReLU(W1 h_k)                          # compress 1024 -> hidden_dim
    a_k  = w^T ( tanh(V h'_k) * sigmoid(U h'_k) )  # one raw score per tile
    alpha = softmax(a_1..a_N)                    # attention: sums to 1 over tiles
    z    = sum_k alpha_k h'_k                    # slide embedding (hidden_dim,)
    logits = classifier(z)

`alpha` is the heatmap: high alpha = "Iris looked here".
"""
import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256, attn_dim=128, n_classes=6, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(hidden_dim, attn_dim)   # the tanh branch
        self.attn_U = nn.Linear(hidden_dim, attn_dim)   # the sigmoid "gate" branch
        self.attn_w = nn.Linear(attn_dim, 1)            # attn_dim -> one score per tile
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor) -> dict:
        """h: (N, feat_dim) -> {"logits": (1, n_classes), "attention": (N,)}"""
        # TODO 1: h' = self.fc(h)                                   -> (N, hidden_dim)
        # TODO 2: raw scores a = attn_w( tanh(V h') * sigmoid(U h') ) -> (N, 1)
        # TODO 3: alpha = softmax over the N tiles (careful with the dim!)
        # TODO 4: z = weighted sum of h' with alpha                  -> (hidden_dim,)
        # TODO 5: logits = self.classifier(z), reshape to (1, n_classes)
        # Return the attention too, the heatmap needs it.
        # Check yourself with:  pytest tests/ -k abmil
        raise NotImplementedError("Your turn! See the docstring at the top of abmil.py")

# LATER (Iris v1.5): CLAM adds an "instance-level clustering" loss on the top-k
# and bottom-k attended tiles. Add it only once plain ABMIL works and you can
# measure whether it helps.
