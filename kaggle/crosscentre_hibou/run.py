"""Kaggle GPU run: experiment C at 20x (cross-centre) with Hibou-B features.

Runs scripts/08_cross_centre.py on all 5 folds, 30 epochs, on the GPU. The features come
from earlier runs attached as kernel_sources. The ResNet and Hibou-B versions of this run
differ ONLY in those features. (A CPU test first estimated ~18 h per encoder, too slow.)

Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/crosscentre_hibou
Output: timing.json and runs/<date>_crosscentre/ (results.json, predictions, models).
"""
import json
import os
import shutil
import subprocess
import sys
import time


REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
DATA = "/tmp/data"            # links to the features, not saved as output
OUT = "/kaggle/working"


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


t0 = time.time()
run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)

os.makedirs(f"{DATA}/features", exist_ok=True)
n = 0
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.basename(root) == "features":
        for f in files:
            if f.endswith(".pt") and not os.path.exists(f"{DATA}/features/{f}"):
                os.symlink(os.path.join(root, f), f"{DATA}/features/{f}")
                n += 1
print(f"feature files found: {n}", flush=True)
if n < 9000:
    raise SystemExit(f"expected about 9,022 feature files, found {n}")
for f in ("splits.csv", "test_split.csv"):
    shutil.copy(f"{CODE}/kaggle/splits/{f}", f"{DATA}/{f}")

import torch  # noqa: E402
timing = {"encoder": "hibou", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
          "feature_files": n, "setup_s": round(time.time() - t0)}
env = dict(os.environ, IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0")
t = time.time()
run(sys.executable, f"{CODE}/scripts/08_cross_centre.py", "--device", "cuda", env=env)
timing["run_s"] = round(time.time() - t)
timing["total_s"] = round(time.time() - t0)
shutil.copytree(f"{DATA}/runs", f"{OUT}/runs", dirs_exist_ok=True)
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
