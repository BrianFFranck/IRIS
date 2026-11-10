"""Add a finished run to experiments.md and refresh the lab card on the GitHub profile.

04_train.py calls this by itself after a full cross-validation run, so you only need
it by hand to log an older run, to preview a row, or to write your own note.

Usage:
  python scripts/06_log_run.py                       # newest run folder
  python scripts/06_log_run.py runs/20261110-1707_meanmax --dry-run
  python scripts/06_log_run.py --note "Attention finally separates GG1 from GG2."
  python scripts/06_log_run.py --no-publish          # write the row, don't push
"""
import argparse
from pathlib import Path

from iris.config import PATHS
from iris.runlog import log_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", type=Path, help="run folder (default: the newest one)")
    ap.add_argument("--note", default="", help="your finding, in a sentence or two (default: written from the numbers)")
    ap.add_argument("--settings", default="", help="anything worth recording, e.g. 'boundary-weighted loss'")
    ap.add_argument("--no-publish", action="store_true", help="don't commit, push or refresh the profile card")
    ap.add_argument("--dry-run", action="store_true", help="print the row and change nothing")
    args = ap.parse_args()
    run = args.run or max(p for p in PATHS.runs.iterdir() if p.is_dir())
    row = log_run(run, args.note, args.settings, publish=not args.no_publish, dry_run=args.dry_run)
    if args.dry_run:
        print(row)


if __name__ == "__main__":
    main()
