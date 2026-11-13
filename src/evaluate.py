"""Inference and evaluation on the fixed test set.

Can be imported (used by train.py) or run on its own to evaluate a saved
checkpoint:

    python src/evaluate.py --ckpt checkpoints/rq1_n100_s0.pt --data_root <DATA_ROOT>
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from common import CLASSES, ImageListDataset, build_model, get_device


@torch.no_grad()
def predict(model, loader, device):
    """Run inference and return (predictions, ground-truth labels)."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def compute_metrics(preds, labels):
    """Accuracy, macro-F1, per-class precision/recall/F1 and confusion matrix."""
    idx = list(range(len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=idx, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=idx, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=idx).tolist(),
    }


def make_test_loader(test_csv, data_root, batch_size=64, num_workers=4):
    test_ds = ImageListDataset(pd.read_csv(test_csv), data_root, train=False)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_csv", default="splits/test.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device(args.device)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.to(device)

    loader = make_test_loader(args.test_csv, args.data_root, num_workers=args.num_workers)
    metrics = compute_metrics(*predict(model, loader, device))
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print("Confusion matrix (rows = true, cols = predicted):", CLASSES)
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
