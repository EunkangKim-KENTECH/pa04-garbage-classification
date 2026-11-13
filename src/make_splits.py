"""Create the fixed test set and the training pool.

The test set is balanced (--test_per_class images per class) and is shared
by ALL experiments. The remaining images form the pool from which each
experiment samples its own training set (see train.py).
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import CLASSES

IMG_EXTS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Folder that contains one sub-folder per class")
    parser.add_argument("--out_dir", default="splits")
    parser.add_argument("--test_per_class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    test_rows, pool_rows = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(args.data_root, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS))
        files = rng.permutation(files).tolist()
        test_rows += [(f"{cls}/{f}", cls) for f in files[:args.test_per_class]]
        pool_rows += [(f"{cls}/{f}", cls) for f in files[args.test_per_class:]]

    os.makedirs(args.out_dir, exist_ok=True)
    test_df = pd.DataFrame(test_rows, columns=["path", "label"])
    pool_df = pd.DataFrame(pool_rows, columns=["path", "label"])
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    pool_df.to_csv(os.path.join(args.out_dir, "pool.csv"), index=False)

    summary = pd.DataFrame({
        "test": test_df["label"].value_counts(),
        "pool": pool_df["label"].value_counts(),
    }).reindex(CLASSES)
    summary["total"] = summary["test"] + summary["pool"]
    print(summary.to_string())
    print(f"Total: test={len(test_df)}, pool={len(pool_df)}")


if __name__ == "__main__":
    main()
