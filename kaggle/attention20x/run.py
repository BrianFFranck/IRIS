"""Kaggle CPU run: attention check at 20x, ResNet-50 vs Hibou-B, same slides and same tiles.

Runs scripts/11_attention_check.py on the Radboud GG2 development slides. Inputs (attached in
kernel-metadata.json): the PANDA label masks, the 20x ResNet-50 features (iris-20x-full and
chunk2..4) and Hibou-B features (iris-hibou-part1/2) of the same tiles, and the trained fold
models (private dataset imatlas/iris-models: resnet20x = 20261115-1401_abmil, hibou20x =
20261122-1942_abmil). No GPU needed.

Send from the laptop with:  .venv/bin/kaggle kernels push -p kaggle/attention20x
Output: attention_check.csv
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
DATA = "/tmp/data"
OUT = "/kaggle/working"
SKIP = ("train_images", "train_label_masks")


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def find_panda(top="/kaggle/input"):
    for root, dirs, files in os.walk(top):
        if "train.csv" in files and "train_label_masks" in dirs:
            return root
        dirs[:] = [d for d in dirs if d not in SKIP + ("features",)]
    raise SystemExit(f"PANDA not found under {top}")


def find_models(name, top="/kaggle/input"):
    for root, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if d not in SKIP + ("features",)]
        if os.path.basename(root) == name and os.path.exists(f"{root}/fold0/best.pt"):
            return root
    raise SystemExit(f"models '{name}' not found: is the dataset imatlas/iris-models attached?")


run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)

# link each encoder's feature files into its own folder (both are called features/ in the inputs)
for enc in ("resnet", "hibou"):
    os.makedirs(f"{DATA}/{enc}", exist_ok=True)
for root, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if d not in SKIP]
    if os.path.basename(root) == "features":
        enc = "hibou" if "hibou" in root else "resnet"
        for f in files:
            if f.endswith(".pt") and not os.path.exists(f"{DATA}/{enc}/{f}"):
                os.symlink(os.path.join(root, f), f"{DATA}/{enc}/{f}")
for enc in ("resnet", "hibou"):
    n = len(os.listdir(f"{DATA}/{enc}"))
    print(f"{enc}: {n} feature files", flush=True)
    if n < 9000:
        raise SystemExit(f"expected about 9,021 {enc} feature files, found {n}")
for f in ("splits.csv", "test_split.csv"):
    shutil.copy(f"{CODE}/kaggle/splits/{f}", f"{DATA}/{f}")

env = dict(os.environ, IRIS_PANDA_DIR=find_panda(), IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0")
run(sys.executable, f"{CODE}/scripts/11_attention_check.py",
    "--encoder", "resnet", f"{DATA}/resnet", find_models("resnet20x"),
    "--encoder", "hibou", f"{DATA}/hibou", find_models("hibou20x"),
    "--out", OUT, env=env)
