"""Kaggle CPU run: which slides got the reference stain colours (the fallback in iris/data/stain.py)?

Reads the flag `stain_reference_colours` that 03_extract_features.py --stain macenko saved in
every feature file, plus each slide's number of tiles. The analysis runs on the laptop.

Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/stain_fallback
Output: stain_fallback.csv (image_id, reference_colours, n_tiles).
"""
import csv
import os

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

with open("/kaggle/working/stain_fallback.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["image_id", "reference_colours", "n_tiles"])
    for i, (image_id, path) in enumerate(sorted(paths.items())):
        bag = torch.load(path, weights_only=False)
        w.writerow([image_id, int(bool(bag["stain_reference_colours"])), len(bag["features"])])
        if i % 1000 == 0:
            print(i, flush=True)
print("done", len(paths))
