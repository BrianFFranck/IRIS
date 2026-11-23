"""Kaggle training run, leakage check step B: Hibou-B at ~20x with folds that keep similar slides together.

Exactly the same as kaggle/trainhibou (2026-11-22, QWK 0.88), except the folds: this run uses
kaggle/splits/splits_grouped.csv (made by scripts/10_grouped_folds.py), where slides that look
like close twins (probably the same patient) are never split across folds. Download with
  kaggle kernels output imatlas/iris-train-grouped -p runs_kaggle/traingrouped
The rule is in experiments.md (2026-11-23).
"""
import os
import shutil
import subprocess
import sys
import time

MODEL = "abmil"
EXTRA = []   # plain cross-entropy, exactly as the ResNet 20x baseline

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
shutil.copy(f"{CODE}/kaggle/splits/splits_grouped.csv", f"{DATA}/splits.csv")   # the grouped folds
shutil.copy(f"{CODE}/kaggle/splits/test_split.csv", f"{DATA}/test_split.csv")

env = dict(os.environ, IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0")
run(sys.executable, f"{CODE}/scripts/04_train.py", "--model", MODEL, "--device", "cuda", "--no-log", *EXTRA, env=env)

shutil.copytree(f"{DATA}/runs", f"{OUT}/runs", dirs_exist_ok=True)
print(f"done in {round((time.time() - t0) / 60)} min", flush=True)
