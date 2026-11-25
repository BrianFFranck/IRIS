"""Kaggle training run: gated attention MIL on the 9,022 development slides at ~20x, Hibou-B features
of stain-normalised tiles (Macenko, see iris/data/stain.py).

Reads the feature files made by the featurise notebooks (attached as kernel_sources in
kernel-metadata.json), uses the SAME folds as on the laptop (kaggle/splits/), trains all
5 folds on the GPU, and saves the run folder (history, validation predictions, weights)
to the output. Download it with
  kaggle kernels output imatlas/iris-train-hiboustain -p runs_kaggle/trainhiboustain
and log it on the laptop with scripts/06_log_run.py.

Fair comparison: this run and trainhibou (Hibou-B, 2026-11-22) differ ONLY in the tile colours:
same tiles, same encoder, same folds, same model and settings. The rule is in experiments.md (2026-11-25).
"""
import os
import shutil
import subprocess
import sys
import time

MODEL = "abmil"
EXTRA = []   # plain cross-entropy, exactly as trainhibou

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
DATA = "/tmp/data"            # not /kaggle/working, so 9 GB of links are not saved as output
OUT = "/kaggle/working"


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


t0 = time.time()
run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)

# gather every feature file from the attached notebooks into one folder (as links)
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
    raise SystemExit(f"expected about 9,022 feature files, found {n}: is every featurise notebook attached and finished?")
for f in ("splits.csv", "test_split.csv"):
    shutil.copy(f"{CODE}/kaggle/splits/{f}", f"{DATA}/{f}")

env = dict(os.environ, IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0")
run(sys.executable, f"{CODE}/scripts/04_train.py", "--model", MODEL, "--device", "cuda", "--no-log", *EXTRA, env=env)

shutil.copytree(f"{DATA}/runs", f"{OUT}/runs", dirs_exist_ok=True)
print(f"done in {round((time.time() - t0) / 60)} min", flush=True)
