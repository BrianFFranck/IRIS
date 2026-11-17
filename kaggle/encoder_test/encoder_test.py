"""Kaggle test run: try two pathology encoders on the same 100 slides at ~20x.

Before choosing an encoder for all 9,022 development slides, measure what it really
costs: time per slide, disk space, and whether it loads at all on Kaggle.

  hibou-b   HistAI Hibou-B (Apache 2.0), small. Gated: needs a Kaggle secret HF_TOKEN
            (a Hugging Face token whose account accepted Hibou-B's terms); skipped without it.
  midnight  kaiko.ai Midnight-12k (MIT), large. Not gated.

Sent from the laptop with:   .venv/bin/kaggle kernels push -p kaggle/encoder_test
Results downloaded with:     .venv/bin/kaggle kernels output imatlas/iris-encoder-test -p data/kaggle_encoder_test

It installs Iris from GitHub, so commit and push before sending it.
Output (in /kaggle/working): coords/*.npy, features_<encoder>/*.pt and timing.json.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
OUT = Path("/kaggle/working")
ENCODERS = ["hibou-b"]   # v2: Hibou-B only (v1 measured Midnight: 40 s per 1,000 tiles on a T4, 296 MB per 100 slides)


def run(*cmd, env=None, check=True):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=check, env=env).returncode


def find_panda(top="/kaggle/input"):
    """Kaggle has moved its data folders before, so look for PANDA instead of assuming a path."""
    for root, dirs, files in os.walk(top):
        if "train.csv" in files and "train_images" in dirs:
            return root
        dirs[:] = [d for d in dirs if d not in ("train_images", "train_label_masks")]
    raise SystemExit(f"PANDA not found under {top}: {os.listdir(top)}")


def hf_token():
    """The Hugging Face token, if it was added as a Kaggle secret called HF_TOKEN."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"no HF_TOKEN secret ({type(e).__name__}): gated encoders will be skipped", flush=True)
        return None


t0 = time.time()
run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)
env = dict(os.environ,
           IRIS_PANDA_DIR=find_panda(),
           IRIS_DATA_DIR=str(OUT),
           IRIS_TILE_LEVEL="0")          # level 0 = full resolution, ~20x
token = hf_token()
if token:
    env["HF_TOKEN"] = token

import torch                              # noqa: E402  (after the install, like the scripts)
timing = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
          "setup_s": round(time.time() - t0)}

t = time.time()
run(sys.executable, f"{CODE}/scripts/02_tile_slides.py", "--ids", f"{CODE}/kaggle/ids_100.txt", env=env)
timing["tiling_s"] = round(time.time() - t)
coords = sorted((OUT / "coords").glob("*.npy"))
timing["slides"] = len(coords)
timing["tiles"] = int(sum(len(__import__("numpy").load(c)) for c in coords))
print(f"slides tiled: {timing['slides']}, tiles: {timing['tiles']}", flush=True)
if not coords:
    raise SystemExit("no slides were tiled: stopping before feature extraction")

for enc in ENCODERS:
    out = OUT / f"features_{enc}"
    t = time.time()
    code = run(sys.executable, f"{CODE}/scripts/03_extract_features.py", "--device", "cuda",
               "--encoder", enc, "--out", out, env=env, check=False)
    bags = sorted(out.glob("*.pt")) if out.is_dir() else []
    res = {"ok": code == 0 and len(bags) == len(coords), "exit_code": code,
           "seconds": round(time.time() - t), "bags": len(bags),
           "megabytes": round(sum(b.stat().st_size for b in bags) / 1e6, 1)}
    if bags:
        res["feat_dim"] = int(torch.load(bags[0], weights_only=True)["features"].shape[1])
        res["seconds_per_1000_tiles"] = round(res["seconds"] / timing["tiles"] * 1000, 1)
    timing[enc] = res
    print(enc, res, flush=True)
    json.dump(timing, open(OUT / "timing.json", "w"), indent=1)   # saved after each encoder

timing["total_s"] = round(time.time() - t0)
json.dump(timing, open(OUT / "timing.json", "w"), indent=1)
print(json.dumps(timing, indent=1))
