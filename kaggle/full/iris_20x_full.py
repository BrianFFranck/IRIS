"""Kaggle run: tile and featurise CHUNK 1 of the 9,022 development slides at ~20x.

Chunks 2-4 and the 5x run have their own folders (kaggle/chunk2 ... kaggle/5x_all), one
Kaggle notebook each, because Kaggle keeps only the output of a notebook's latest run.
Do NOT push this folder again: a new version would replace chunk 1's features.

The held-out test slides are in none of these lists: they are processed once, at the
end, with the final pipeline. This folder also holds the slide lists (ids_chunk*.txt,
ids_all.txt) that every run reads from GitHub.
"""
import json
import os
import subprocess
import sys
import time

CHUNK = 1                     # 1, 2, 3 or 4: change this before each push

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
env = dict(os.environ, IRIS_PANDA_DIR=find_panda(), IRIS_DATA_DIR=OUT, IRIS_TILE_LEVEL="0")
ids = f"{CODE}/kaggle/full/ids_chunk{CHUNK}.txt"
print(f"chunk {CHUNK}: {sum(1 for _ in open(ids))} slides; PANDA at {env['IRIS_PANDA_DIR']}", flush=True)
timing = {"chunk": CHUNK, "setup_s": round(time.time() - t0)}

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
