"""Kaggle GPU run: Hibou-B features of the PAR slides (Erbil, Iraq; Grundium scanner), part 2 of 2.

External validation: PAR is never used for training. Slides are streamed from the EBI server
(BioImage Archive S-BIAD2323, CC BY 4.0): downloaded a few at a time, cut into tiles by microns
so the zoom matches PANDA 20x, turned into features in two colour versions (original and
Macenko), and deleted (scripts/13_external_features.py). See docs/external_datasets.md.

Hibou-B is gated, so this run needs the Kaggle secret HF_TOKEN: push once with
  kaggle kernels push -p kaggle/par2
(that run stops at once, without the token), then on the website: Edit -> HF_TOKEN ticked ->
Save Version -> Save & Run All (Commit).
Output: /kaggle/working/par/original/*.pt, /kaggle/working/par/macenko/*.pt, timing.json.
"""
import json
import os
import subprocess
import sys
import time

PART = 2                      # part 1: slides 1-170 of kaggle/par/grundium_urls.txt; part 2: 171-339

REPO = "https://github.com/BrianFFranck/IRIS.git"
CODE = "/tmp/IRIS"
OUT = "/kaggle/working"


def run(*cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


t0 = time.time()
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    raise SystemExit(f"no HF_TOKEN secret ({type(e).__name__}): start this run from the website "
                     "(Edit -> HF_TOKEN ticked -> Save Version). Stopping before any work.")

run("git", "clone", "--depth", "1", REPO, CODE)
run(sys.executable, "-m", "pip", "install", "-q", CODE)

urls = [l.strip() for l in open(f"{CODE}/kaggle/par/grundium_urls.txt") if l.strip()]
mine = urls[:170] if PART == 1 else urls[170:]
open("/tmp/my_slides.txt", "w").write("\n".join(mine) + "\n")
print(f"part {PART}: {len(mine)} slides", flush=True)

env = dict(os.environ, HF_TOKEN=token)
t = time.time()
run(sys.executable, f"{CODE}/scripts/13_external_features.py", "--list", "/tmp/my_slides.txt",
    "--out", f"{OUT}/par", "--stain", "none", "macenko", "--encoder", "hibou-b", "--device", "cuda",
    "--workers", "4", "--prefetch", "3", "--hours", "10.5", env=env)
timing = {"part": PART, "slides": len(mine), "features_s": round(time.time() - t),
          "done_original": len(os.listdir(f"{OUT}/par/original")), "done_macenko": len(os.listdir(f"{OUT}/par/macenko")),
          "total_s": round(time.time() - t0)}
json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print(timing)
