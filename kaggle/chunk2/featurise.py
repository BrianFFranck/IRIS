"""Kaggle run: tile and featurise chunk 2 of the development slides at ~20x.

One notebook per run, so every output stays available for the training notebook.
Send from the laptop with:  kaggle kernels push -p kaggle/chunk2
"""
import json
import os
import subprocess
import sys
import time

IDS = "kaggle/full/ids_chunk2.txt"
LEVEL = "0"             # "0" = ~20x, "1" = ~5x

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
OUT = "/kaggle/working"


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def find_panda(top="/kaggle/input"):
    """Kaggle has moved its data folders before, so look for PANDA instead of assuming a path."""
    for root, dirs, files in os.walk(top):
        if "train.csv" in files and "train_images" in dirs:
            return root
        dirs[:] = [d for d in dirs if d not in ("train_images", "train_label_masks")]
    raise SystemExit(f"PANDA not found under {top}: {os.listdir(top)}")


t0 = time.time()
run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)
env = dict(os.environ, IRIS_PANDA_DIR=find_panda(), IRIS_DATA_DIR=OUT, IRIS_TILE_LEVEL=LEVEL)
ids = f"{CODE}/{IDS}"
print(f"{IDS} at level {LEVEL}: {sum(1 for _ in open(ids))} slides; PANDA at {env['IRIS_PANDA_DIR']}", flush=True)
timing = {"ids": IDS, "level": LEVEL, "setup_s": round(time.time() - t0)}

t = time.time()
run(sys.executable, f"{CODE}/scripts/02_tile_slides.py", "--ids", ids, env=env)
timing["tiling_s"] = round(time.time() - t)
n_tiled = len(os.listdir(f"{OUT}/coords")) if os.path.isdir(f"{OUT}/coords") else 0
print(f"slides tiled: {n_tiled}", flush=True)
if n_tiled == 0:
    raise SystemExit("no slides were tiled: stopping before feature extraction")

t = time.time()
run(sys.executable, f"{CODE}/scripts/03_extract_features.py", "--device", "cuda", "--workers", "4", env=env)
timing["features_s"] = round(time.time() - t)

timing["slides"] = n_tiled
timing["total_s"] = round(time.time() - t0)
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
