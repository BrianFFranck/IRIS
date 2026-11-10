"""Download train.csv + a SMALL stratified subset of PANDA slides for local dev.

The full PANDA train set is ~400 GB (10,616 slides)

Usage:
  python scripts/01_download_subset.py --n 300
  python scripts/01_download_subset.py --n 300 --masks   # + pixel-level label masks
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from iris.config import PATHS

COMP = "prostate-cancer-grade-assessment"

# Use the kaggle CLI from the same environment as this Python, so the script
# works even when the venv is not activated (e.g. `.venv/bin/python scripts/...`).
_venv_kaggle = Path(sys.executable).parent / "kaggle"
KAGGLE = str(_venv_kaggle) if _venv_kaggle.exists() else shutil.which("kaggle")
if KAGGLE is None:
    sys.exit("kaggle CLI not found: install it with  uv pip install kaggle")


def kaggle_get(file: str):
    """Download one competition file into data/panda/, keeping its sub-folder
    (e.g. train_images/<id>.tiff). The kaggle CLI recreates that sub-folder
    itself and often delivers the file zipped, so unzip whatever arrives."""
    PATHS.panda.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "competitions", "download", "-c", COMP, "-f", file, "-p", str(PATHS.panda), "-q"], check=True)
    unzip_all()


def unzip_all():
    for z in PATHS.panda.rglob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)
        z.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of slides")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--masks", action="store_true", help="also download the label masks of the slides")
    args = ap.parse_args()

    if not PATHS.train_csv.exists():
        kaggle_get("train.csv")
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
        kaggle_get(f"train_images/{image_id}.tiff")

    if args.masks:
        download_masks()


def download_masks():
    """Pixel-level label masks for every slide already in train_images/.
    Not every PANDA slide has a mask, so a missing one is reported, not fatal."""
    ids = sorted(p.stem for p in PATHS.images.glob("*.tiff"))
    missing = []
    print(f"downloading label masks for {len(ids)} slides -> {PATHS.masks}")
    for i, image_id in enumerate(ids, 1):
        if (PATHS.masks / f"{image_id}_mask.tiff").exists():
            continue
        print(f"[{i}/{len(ids)}] {image_id}_mask")
        try:
            kaggle_get(f"train_label_masks/{image_id}_mask.tiff")
        except subprocess.CalledProcessError:
            missing.append(image_id)
    if missing:
        print(f"{len(missing)} slides have no label mask on Kaggle: {missing}")


if __name__ == "__main__":
    main()
