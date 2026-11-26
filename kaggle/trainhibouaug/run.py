"""Kaggle GPU run: the main 5-fold Hibou-B model at 20x with stain AUGMENTATION.

Training slides are read either with their original colours or with one random colouring
(50/50, from kaggle/hibouaug1/2, see iris/data/stain.py); validation and test slides always
keep their original colours, so nothing is repainted at test time. Compared with
kaggle/trainhibou (same model without augmentation) using scripts/12_compare_runs.py.
The rule is in experiments.md.

Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/trainhibouaug
Output: timing.json and runs/<date>_abmil/ (history, validation predictions, weights).
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


def gather(name):
    """Link every .pt file from the attached notebooks' folders called `name` into DATA/name."""
    os.makedirs(f"{DATA}/{name}", exist_ok=True)
    n = 0
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.basename(root) == name:
            for f in files:
                if f.endswith(".pt") and not os.path.exists(f"{DATA}/{name}/{f}"):
                    os.symlink(os.path.join(root, f), f"{DATA}/{name}/{f}")
                    n += 1
    print(f"{name}: {n} files", flush=True)
    if n < 9000:
        raise SystemExit(f"expected about 9,021 files in {name}, found {n}: are all four notebooks attached and finished?")
    return n


t0 = time.time()
run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)
n = gather("features")
n_aug = gather("features_aug")
for f in ("splits.csv", "test_split.csv"):
    shutil.copy(f"{CODE}/kaggle/splits/{f}", f"{DATA}/{f}")

import torch  # noqa: E402
timing = {"encoder": "hibou", "stain_augmentation": True, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
          "feature_files": n, "augmented_files": n_aug, "setup_s": round(time.time() - t0)}
env = dict(os.environ, IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0")
t = time.time()
run(sys.executable, f"{CODE}/scripts/04_train.py", "--model", "abmil", "--device", "cuda", "--no-log",
    "--aug-features", f"{DATA}/features_aug", env=env)
timing["run_s"] = round(time.time() - t)
timing["total_s"] = round(time.time() - t0)
shutil.copytree(f"{DATA}/runs", f"{OUT}/runs", dirs_exist_ok=True)
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
