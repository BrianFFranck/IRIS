"""Kaggle CPU run: one summary vector per development slide, for the leakage check.

Each slide's vector = the average of its Hibou-B tile features (768 numbers), plus its
number of tiles. The analysis itself runs on the laptop (scripts/09_leakage_check.py).

Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/slide_embeddings
Output: slide_embeddings.npz (image_id, mean, n_tiles).
"""
import os

import numpy as np
import torch

paths = {}
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.basename(root) == "features":
        for f in files:
            if f.endswith(".pt"):
                paths.setdefault(f[:-3], os.path.join(root, f))
print(f"feature files found: {len(paths)}", flush=True)
if len(paths) < 9000:
    raise SystemExit(f"expected about 9,021 feature files, found {len(paths)}")

ids, means, counts = [], [], []
for i, (image_id, path) in enumerate(sorted(paths.items())):
    f = torch.load(path, weights_only=True)["features"].float()
    ids.append(image_id)
    means.append(f.mean(0).numpy())
    counts.append(len(f))
    if i % 1000 == 0:
        print(i, flush=True)
np.savez("/kaggle/working/slide_embeddings.npz", image_id=np.array(ids),
         mean=np.stack(means).astype(np.float32), n_tiles=np.array(counts))
print("done", len(ids))
