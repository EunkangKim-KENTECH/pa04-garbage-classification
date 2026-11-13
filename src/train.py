"""Train one ResNet-18 classifier for one experimental condition.

The training set is sampled from splits/pool.csv:
  * --per_class N : take N images from every class
  * --trash T     : (optional) override the number of `trash` images (RQ2)

Every run uses the SAME number of optimisation steps (--steps), so that
conditions differ only in the training data, not in the training budget.
After training, the model is evaluated once on the fixed test set and the
results are written to <runs_dir>/<run_name>/metrics.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from common import CLASSES, ImageListDataset, build_model, get_device, set_seed
from evaluate import compute_metrics, make_test_loader, predict


def sample_train_set(pool_df, counts, seed):
    """Sample `counts[cls]` images per class from the pool (without replacement)."""
    rng = np.random.RandomState(seed)
    parts = []
    for cls in CLASSES:
        cls_df = pool_df[pool_df["label"] == cls].sort_values("path")
        n = counts[cls]
        if n > len(cls_df):
            raise ValueError(f"Requested {n} '{cls}' images but pool has {len(cls_df)}")
        parts.append(cls_df.iloc[rng.choice(len(cls_df), size=n, replace=False)])
    return pd.concat(parts).reset_index(drop=True)


def class_weights(counts, device):
    """Inverse-frequency weights: w_c = N_total / (num_classes * N_c)."""
    n = torch.tensor([counts[c] for c in CLASSES], dtype=torch.float)
    return (n.sum() / (len(CLASSES) * n)).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--per_class", type=int, required=True)
    parser.add_argument("--trash", type=int, default=None,
                        help="Override the number of trash images (class imbalance)")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Use class-weighted cross-entropy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pool_csv", default="splits/pool.csv")
    parser.add_argument("--test_csv", default="splits/test.csv")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Random init (only for quick debugging)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Data ----
    counts = {c: args.per_class for c in CLASSES}
    if args.trash is not None:
        counts["trash"] = args.trash
    train_df = sample_train_set(pd.read_csv(args.pool_csv), counts, args.seed)
    train_df.to_csv(os.path.join(run_dir, "train_list.csv"), index=False)

    train_ds = ImageListDataset(train_df, args.data_root, train=True)
    g = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(train_ds, replacement=True,
                            num_samples=args.steps * args.batch_size, generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = make_test_loader(args.test_csv, args.data_root, num_workers=args.num_workers)

    # ---- Model / optimiser ----
    model = build_model(pretrained=not args.no_pretrained).to(device)
    weight = class_weights(counts, device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Train (fixed number of steps) ----
    model.train()
    start = time.time()
    for step, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            print(f"[{args.run_name}] step {step}/{args.steps} loss {loss.item():.4f}", flush=True)
    train_time = time.time() - start

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.run_name}.pt"))

    # ---- Evaluate on the fixed test set ----
    metrics = compute_metrics(*predict(model, test_loader, device))
    metrics.update({
        "run_name": args.run_name,
        "config": vars(args),
        "train_counts": counts,
        "train_time_sec": round(train_time, 1),
        "device": str(device),
    })
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{args.run_name}] acc={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} "
          f"trash_recall={metrics['per_class']['trash']['recall']:.4f}")


if __name__ == "__main__":
    main()
