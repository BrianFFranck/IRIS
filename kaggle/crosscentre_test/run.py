"""Kaggle CPU speed test for experiment C at 20x (cross-centre), before the full runs.

Runs scripts/08_cross_centre.py on the Hibou-B features (attached as kernel_sources) for
ONE fold and a few epochs, on the CPU (no GPU quota used), and measures how long it takes,
so the full run (5 folds x 2 centres x 30 epochs) can be estimated before it is started.

Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/crosscentre_test
Output: timing.json and the run folder.
"""
import json
import os
import shutil
import subprocess
import sys
import time

FOLDS = ["0"]
EPOCHS = "2"

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
timing = {"cpu_threads": torch.get_num_threads(), "feature_files": n, "folds": FOLDS,
          "epochs": int(EPOCHS), "setup_s": round(time.time() - t0)}
env = dict(os.environ, IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0")
t = time.time()
run(sys.executable, f"{CODE}/scripts/08_cross_centre.py", "--device", "cpu",
    "--folds", *FOLDS, "--epochs", EPOCHS, env=env)
timing["run_s"] = round(time.time() - t)
# 2 models (one per centre) x EPOCHS epochs, plus scoring; the full run is 10 models x 30 epochs
timing["estimate_full_h"] = round(timing["run_s"] / (len(FOLDS) * int(EPOCHS)) * 5 * 30 / 3600, 1)
timing["total_s"] = round(time.time() - t0)
shutil.copytree(f"{DATA}/runs", f"{OUT}/runs", dirs_exist_ok=True)
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
