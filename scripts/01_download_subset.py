"""Download train.csv + a SMALL stratified subset of PANDA slides for local dev.

The full PANDA train set is ~400 GB (10,616 slides): it does not fit on this
laptop and you do not need it to develop. ~300 slides (~10-15 GB) is plenty to
build and debug every step. The full run happens on Kaggle (see the guide).

Setup (once):
  1. kaggle.com -> join the "Prostate cANcer graDe Assessment (PANDA)" competition
     and accept its rules (otherwise downloads give 403).
  2. kaggle.com -> Settings -> API -> "Create New Token" -> saves kaggle.json
  3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Usage:
  python scripts/01_download_subset.py --n 300
"""
import argparse
import subprocess
import zipfile

import pandas as pd

from iris.config import PATHS

COMP = "prostate-cancer-grade-assessment"


def kaggle_get(file: str, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-f", file, "-p", str(dest), "-q"], check=True)
    for z in dest.glob("*.zip"):                 # kaggle sometimes zips single files
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        z.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of slides")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not PATHS.train_csv.exists():
        kaggle_get("train.csv", PATHS.panda)
    df = pd.read_csv(PATHS.train_csv)
    print(df.groupby(["data_provider", "isup_grade"]).size().unstack(fill_value=0))

    # same proportion of every (provider, grade) group as the full dataset
    frac = args.n / len(df)
    sub = df.groupby(["data_provider", "isup_grade"], group_keys=False).sample(frac=frac, random_state=args.seed)
    print(f"downloading {len(sub)} slides -> {PATHS.images}")

    for i, image_id in enumerate(sub.image_id, 1):
        if (PATHS.images / f"{image_id}.tiff").exists():
            continue
        print(f"[{i}/{len(sub)}] {image_id}")
        kaggle_get(f"train_images/{image_id}.tiff", PATHS.images)


if __name__ == "__main__":
    main()
