"""Kaggle test run: tile and featurise 100 of my development slides at ~20x (level 0).

Sent from the laptop with:   kaggle kernels push -p kaggle
Results downloaded with:     kaggle kernels output imatlas/iris-20x-test -p data/kaggle_20x_test

Kaggle runs this on its GPU with the PANDA data attached. It installs Iris from
GitHub, so commit and push before sending it. Output (in /kaggle/working):
coords/*.npy, features/*.pt and timing.json (how long each step took).
"""
import json
import os
import subprocess
import sys
import time

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
OUT = "/kaggle/working"


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


t0 = time.time()
run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)
env = dict(os.environ,
           IRIS_PANDA_DIR="/kaggle/input/prostate-cancer-grade-assessment",
           IRIS_DATA_DIR=OUT,
           IRIS_TILE_LEVEL="0")          # level 0 = full resolution, ~20x
timing = {"setup_s": round(time.time() - t0)}

t = time.time()
run(sys.executable, f"{CODE}/scripts/02_tile_slides.py", "--ids", f"{CODE}/kaggle/ids_100.txt", env=env)
timing["tiling_s"] = round(time.time() - t)

t = time.time()
run(sys.executable, f"{CODE}/scripts/03_extract_features.py", "--device", "cuda", env=env)
timing["features_s"] = round(time.time() - t)

timing["total_s"] = round(time.time() - t0)
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
