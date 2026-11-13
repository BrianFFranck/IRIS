"""Training loop for one fold. Works for any model in iris.models.MODELS.

The loop is finished. The TODOs mark the experiments worth trying once it runs.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from iris.config import TrainConfig
from iris.metrics import qwk


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    losses, y_true, y_pred = [], [], []
    for feats, label, _ in tqdm(loader, leave=False, desc="train" if train else "val"):
        feats, label = feats.squeeze(0).to(device), label.to(device)   # batch_size=1
        with torch.set_grad_enabled(train):
            out = model(feats)
            loss = loss_fn(out["logits"], label)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        y_true.append(label.item())
        y_pred.append(out["logits"].argmax(-1).item())
    return float(np.mean(losses)), qwk(y_true, y_pred), y_true, y_pred


def train_fold(model, train_ds, val_ds, cfg: TrainConfig, out_dir: Path, device="cpu"):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    model.to(device)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # TODO (experiment): ISUP is ORDINAL (0<1<...<5) but CrossEntropy treats it as
    # 6 unrelated classes. Try regression (1 output + MSELoss, then round /
    # optimise thresholds), which most top PANDA solutions used, and compare QWK.
    weight = None
    if cfg.class_weights:
        counts = np.bincount(train_ds.df.isup_grade, minlength=cfg.n_classes)
        weight = torch.tensor(counts.sum() / (cfg.n_classes * counts),                              dtype=torch.float32)
        
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best, history = -1.0, []
    for epoch in range(cfg.epochs):
        tr_loss, tr_qwk, *_ = run_epoch(model, train_dl, loss_fn, optimizer, device)
        va_loss, va_qwk, yt, yp = run_epoch(model, val_dl, loss_fn, None, device)
        scheduler.step()
        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_qwk=tr_qwk, va_loss=va_loss, va_qwk=va_qwk))
        print(f"ep {epoch:02d} | train loss {tr_loss:.3f} qwk {tr_qwk:.3f} | val loss {va_loss:.3f} qwk {va_qwk:.3f}")
        if va_qwk > best:
            best = va_qwk
            torch.save(model.state_dict(), out_dir / "best.pt")
            json.dump({"y_true": yt, "y_pred": yp}, open(out_dir / "val_preds.json", "w"))
        # TODO: early stopping (stop if val QWK hasn't improved for ~8 epochs)

    json.dump(history, open(out_dir / "history.json", "w"), indent=1)
    return best
