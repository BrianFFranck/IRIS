"""Kaggle run: Hibou-B features for half of the 9,022 development slides at ~20x.

The tiles are NOT cut again: the tile positions (coords) come from the 20x ResNet runs
(iris-20x-full and iris-20x-chunk2..4, attached as kernel_sources), so the Hibou-B
features describe exactly the same tiles. Only the encoder changes.

Hibou-B is gated, so this run needs the Kaggle secret HF_TOKEN (a Hugging Face token whose
account accepted Hibou-B's terms). Kaggle only hands secrets to runs started from the
website, so: push once with  kaggle kernels push -p kaggle/hibou2  (that run stops at
once, without the token), then on the website: Edit -> HF_TOKEN ticked -> Save Version
-> Save & Run All (Commit).  Output: /kaggle/working/features/*.pt and timing.json.
"""
import json
import os
import subprocess
import sys
import time

CHUNKS = ["3", "4"]          # hibou1: chunks 1+2 (4,512 slides); hibou2: chunks 3+4 (4,510 slides)

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
DATA = "/tmp/data"           # coords links live here, not in the output
OUT = "/kaggle/working"


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def find_panda(top="/kaggle/input"):
    """Kaggle has moved its data folders before, so look for PANDA instead of assuming a path."""
    for root, dirs, files in os.walk(top):
        if "train.csv" in files and "train_images" in dirs:
            return root
        dirs[:] = [d for d in dirs if d not in ("train_images", "train_label_masks", "features")]
    raise SystemExit(f"PANDA not found under {top}: {os.listdir(top)}")


t0 = time.time()
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    raise SystemExit(f"no HF_TOKEN secret ({type(e).__name__}): start this run from the website "
                     "(Edit -> HF_TOKEN ticked -> Save Version). Stopping before any work.")

run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)

# link the coords of this run's slides from the attached 20x notebooks into one folder
ids = set()
for c in CHUNKS:
    ids |= {line.strip() for line in open(f"{CODE}/kaggle/full/ids_chunk{c}.txt") if line.strip()}
os.makedirs(f"{DATA}/coords", exist_ok=True)
for root, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if d not in ("train_images", "train_label_masks", "features")]
    if os.path.basename(root) == "coords":
        for f in files:
            if f.endswith(".npy") and f[:-4] in ids and not os.path.exists(f"{DATA}/coords/{f}"):
                os.symlink(os.path.join(root, f), f"{DATA}/coords/{f}")
n = len(os.listdir(f"{DATA}/coords"))
print(f"chunks {CHUNKS}: {len(ids)} slides listed, coords found for {n}", flush=True)
if n != len(ids):
    raise SystemExit("some coords are missing: is every 20x notebook attached and finished?")

env = dict(os.environ, IRIS_PANDA_DIR=find_panda(), IRIS_DATA_DIR=DATA, IRIS_TILE_LEVEL="0", HF_TOKEN=token)
timing = {"chunks": CHUNKS, "slides": n, "setup_s": round(time.time() - t0)}

t = time.time()
run(sys.executable, f"{CODE}/scripts/03_extract_features.py", "--device", "cuda", "--workers", "4",
    "--encoder", "hibou-b", "--out", f"{OUT}/features", env=env)
timing["features_s"] = round(time.time() - t)
timing["bags"] = len(os.listdir(f"{OUT}/features"))
timing["total_s"] = round(time.time() - t0)
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
