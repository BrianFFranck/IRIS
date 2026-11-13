"""Write a finished run into experiments.md and tell the GitHub profile card about it.

A run folder (runs/<date>_<model>/fold*/) already holds everything needed:
history.json (QWK per epoch) and val_preds.json (predictions at the best epoch).
From those this module builds one table row, puts it at the top of the Models
table. A run only counts as a new best when it beats the best earlier run ON THE
SAME NUMBER OF SLIDES by more than the spread between folds (the larger of the two
stds); a smaller gain is noise. Only a clear new best is pushed automatically and
asks the profile repo to redraw the lab card. Everything else is logged locally and
stays off GitHub until the log is next pushed.

Only experiments.md is ever committed here. Data, weights and run folders stay local.
"""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from iris.config import TileConfig, TrainConfig
from iris.metrics import qwk

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments.md"
PROFILE_REPO = "BrianFFranck/BrianFFranck"
PROFILE_WORKFLOW = "lab-notebook.yml"
MODEL_LABELS = {"meanmax": "meanmax (baseline)", "abmil": "abmil (gated attention MIL)"}
LEVELS = {0: "0 (~20×)", 1: "1 (~5×)", 2: "2 (~1.25×)"}


def summarise(run_dir: Path) -> dict:
    folds = sorted(p for p in run_dir.glob("fold*") if (p / "val_preds.json").exists())
    if not folds:
        raise SystemExit(f"no finished folds in {run_dir}")
    scores, tr_at_best, y_true, y_pred, epochs = [], [], [], [], 0
    for f in folds:
        preds = json.loads((f / "val_preds.json").read_text())
        hist = json.loads((f / "history.json").read_text())
        best = max(hist, key=lambda h: h["va_qwk"])
        scores.append(qwk(preds["y_true"], preds["y_pred"]))
        tr_at_best.append(best["tr_qwk"])
        y_true += preds["y_true"]
        y_pred += preds["y_pred"]
        epochs = max(epochs, len(hist))
    gg2 = [p for t, p in zip(y_true, y_pred) if t == 2]
    return {"folder": run_dir.name, "model": run_dir.name.split("_", 1)[1], "date": run_dir.name[:8],
            "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "train": float(np.mean(tr_at_best)), "slides": len(y_true), "epochs": epochs,
            "gg2_hit": sum(p == 2 for p in gg2), "gg2_n": len(gg2),
            "gg2_miss": Counter(p for p in gg2 if p != 2).most_common(2), "n_folds": len(folds)}


def previous_best(text: str, slides: int):
    """(mean, std) of the best logged run on the same number of slides, or None."""
    runs = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        m = re.search(r"\*\*(\d\.\d+) ± (\d\.\d+)\*\*", line)
        if m and len(cells) > 3 and cells[3].isdigit() and int(cells[3]) == slides:
            runs.append((float(m.group(1)), float(m.group(2))))
    return max(runs) if runs else None


def is_clear_gain(s: dict, best) -> bool:
    """True when the run beats `best` by more than the spread between folds."""
    return best is not None and s["mean"] - best[0] > max(s["std"], best[1])


def auto_note(s: dict, best) -> str:
    parts = []
    if best is None:
        parts.append(f"First run on {s['slides']} slides: nothing to compare with yet.")
    else:
        d, noise = s["mean"] - best[0], max(s["std"], best[1])
        if is_clear_gain(s, best):
            parts.append(f"New best, {d:+.2f} over {best[0]:.2f} on the same slides, larger than the spread between folds (±{noise:.2f}).")
        elif abs(d) <= noise:
            parts.append(f"Within noise of the best so far ({best[0]:.2f}): the difference ({d:+.2f}) is smaller than the spread between folds (±{noise:.2f}).")
        else:
            parts.append(f"Below the best so far ({best[0]:.2f}) by more than the spread between folds.")
    if s["gg2_n"]:
        if s["gg2_hit"] == 0:
            parts.append("Never predicts GG2.")
        if s["gg2_miss"]:
            parts.append("Missed GG2 mostly called " + " or ".join(f"ISUP {g} ({n})" for g, n in s["gg2_miss"]) + ".")
    gap = s["train"] - s["mean"]
    parts.append(f"Train QWK {s['train']:.2f} vs val {s['mean']:.2f}: " +
                 ("overfitting." if gap > 0.15 else "no sign of overfitting."))
    return " ".join(parts)


def make_row(s: dict, note: str, settings: str = "") -> str:
    d = s["date"]
    cfg = TrainConfig()
    setting = f"{s['epochs']} epochs, lr {cfg.lr:g}" + (f", {settings}" if settings else "")
    folds = " / ".join(f"{v:.2f}" for v in s["scores"])
    return (f"| {d[:4]}-{d[4:6]}-{d[6:]} | {MODEL_LABELS.get(s['model'], s['model'])} | {s['slides']} | "
            f"{LEVELS.get(TileConfig().level, TileConfig().level)} | {setting} | "
            f"**{s['mean']:.2f} ± {s['std']:.2f}** (folds {folds}) | {s['gg2_hit']} / {s['gg2_n']} | "
            f"`{s['folder']}` | {note} |")


def insert_row(text: str, row: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("## models"))
    sep = next(i for i in range(start, len(lines)) if lines[i].startswith("|") and set(lines[i]) <= set("|-: "))
    lines.insert(sep + 1, row)
    return "\n".join(lines) + "\n"


def _run(*cmd) -> bool:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"  ! {cmd[0]} is not installed")
        return False
    if r.returncode:
        print(f"  ! {' '.join(cmd[:3])} failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
    return r.returncode == 0


def log_run(run_dir: Path, note: str = "", settings: str = "", publish="auto", dry_run: bool = False) -> str:
    """publish: "auto" = only when the run is a clear new best, True = always, False = never."""
    s = summarise(Path(run_dir))
    text = LOG.read_text(encoding="utf-8")
    best = previous_best(text, s["slides"])
    row = make_row(s, note or auto_note(s, best), settings)
    if publish == "auto":
        publish = is_clear_gain(s, best)
        if not publish:
            print(f"val QWK {s['mean']:.2f} ± {s['std']:.2f} is not a clear new best on {s['slides']} slides: "
                  "logged here only, nothing pushed (use --publish to push anyway)")
    if dry_run:
        return row
    if f"`{s['folder']}`" in text:
        print(f"{s['folder']} is already in experiments.md, nothing added")
    else:
        LOG.write_text(insert_row(text, row), encoding="utf-8")
        print(f"logged {s['folder']}: val QWK {s['mean']:.2f} ± {s['std']:.2f}")
    if publish:
        msg = f"Experiment log: {s['model']} val QWK {s['mean']:.2f} ± {s['std']:.2f}"
        changed = subprocess.run(["git", "status", "--porcelain", "experiments.md"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
        ok = True
        if changed:
            ok = _run("git", "commit", "-q", "-m", msg, "--", "experiments.md") and _run("git", "push", "-q")
        if ok and _run("gh", "workflow", "run", PROFILE_WORKFLOW, "-R", PROFILE_REPO):
            print("profile card is being redrawn (about a minute)")
        else:
            print("not published; the profile card will still pick the log up on its daily refresh once it is pushed")
    return row
