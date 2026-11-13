"""Collect runs/*/metrics.json into result tables (CSV) and figures (PNG + PDF).

Outputs (in --out_dir, default results/):
  rq1_table.csv  RQ1: training-set size vs. performance (mean +- std over seeds)
  rq2_table.csv  RQ2: number of trash images x loss vs. performance
  fig_rq1.*      accuracy / macro-F1 vs. images per class
  fig_rq2.*      trash recall / macro-F1 vs. number of trash images
  fig_cm.*       confusion matrices (balanced vs. most imbalanced)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLASSES

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})


def load_runs(runs_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        cfg = m["config"]
        rows.append({
            "run_name": m["run_name"],
            "exp": m["run_name"].split("_")[0],
            "seed": cfg["seed"],
            "per_class": cfg["per_class"],
            "trash": m["train_counts"]["trash"],
            "loss": "weighted CE" if cfg["weighted_loss"] else "CE",
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "trash_recall": m["per_class"]["trash"]["recall"],
            "trash_precision": m["per_class"]["trash"]["precision"],
            "trash_f1": m["per_class"]["trash"]["f1"],
            "cm": np.array(m["confusion_matrix"]),
        })
    if not rows:
        raise SystemExit(f"No metrics.json found in {runs_dir}")
    return pd.DataFrame(rows)


def mean_std(df, keys, cols):
    g = df.groupby(keys)[cols].agg(["mean", "std"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    g["n_seeds"] = df.groupby(keys).size()
    return g.reset_index()


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_runs(args.runs_dir)
    metric_cols = ["accuracy", "macro_f1", "trash_recall", "trash_precision", "trash_f1"]

    # ---------------- RQ1: data quantity ----------------
    rq1 = df[df["exp"] == "rq1"]
    if len(rq1):
        t1 = mean_std(rq1, ["per_class"], ["accuracy", "macro_f1"])
        t1.to_csv(os.path.join(args.out_dir, "rq1_table.csv"), index=False, float_format="%.4f")
        print("RQ1\n", t1.to_string(index=False))

        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        for col, label, marker in [("accuracy", "Accuracy", "o"), ("macro_f1", "Macro-F1", "s")]:
            ax.errorbar(t1["per_class"], t1[f"{col}_mean"], yerr=t1[f"{col}_std"],
                        marker=marker, capsize=3, label=label)
        ax.set_xscale("log")
        ax.set_xticks(t1["per_class"])
        ax.set_xticklabels(t1["per_class"])
        ax.set_xlabel("Training images per class")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, args.out_dir, "fig_rq1")

    # ---------------- RQ2: class imbalance ----------------
    # The balanced reference point (trash = per_class) comes from RQ1.
    rq2 = df[df["exp"] == "rq2"]
    if len(rq2):
        ref_n = int(rq2["per_class"].iloc[0])
        ref = df[(df["exp"] == "rq1") & (df["per_class"] == ref_n)]
        rq2_all = pd.concat([rq2, ref.assign(loss="CE"), ref.assign(loss="weighted CE")])
        t2 = mean_std(rq2_all, ["trash", "loss"], metric_cols)
        t2.to_csv(os.path.join(args.out_dir, "rq2_table.csv"), index=False, float_format="%.4f")
        print("RQ2\n", t2.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
        for ax, col, title in [(axes[0], "trash_recall", "Recall of the minority class (trash)"),
                               (axes[1], "macro_f1", "Macro-F1 (all classes)")]:
            for loss, marker in [("CE", "o"), ("weighted CE", "s")]:
                sub = t2[t2["loss"] == loss].sort_values("trash")
                ax.errorbar(sub["trash"], sub[f"{col}_mean"], yerr=sub[f"{col}_std"],
                            marker=marker, capsize=3, label=loss)
            ticks = sorted(t2["trash"].unique())
            ax.set_xscale("log")
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)
            ax.set_xlabel(f"Trash training images (other classes: {ref_n})")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()
        save(fig, args.out_dir, "fig_rq2")

        # Confusion matrices summed over seeds: balanced vs. most imbalanced (CE)
        t_min = int(rq2["trash"].min())
        panels = [(ref, f"Balanced ({ref_n} trash)"),
                  (rq2[(rq2["trash"] == t_min) & (rq2["loss"] == "CE")], f"Imbalanced ({t_min} trash, CE)")]
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
        for ax, (sub, title) in zip(axes, panels):
            cm = np.sum(np.stack(sub["cm"].tolist()), axis=0).astype(float)
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if cm_norm[i, j] > 0.5 else "black")
            ax.set_xticks(range(len(CLASSES)))
            ax.set_yticks(range(len(CLASSES)))
            ax.set_xticklabels(CLASSES, rotation=45, ha="right")
            ax.set_yticklabels(CLASSES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
        save(fig, args.out_dir, "fig_cm")

    df.drop(columns=["cm"]).to_csv(os.path.join(args.out_dir, "all_runs.csv"),
                                   index=False, float_format="%.4f")
    print(f"Saved tables and figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
