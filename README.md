# How Much and How Balanced? Data Effects in Recyclable Waste Classification

EF2039 PA04. A ResNet-18 image classifier for six waste categories, used to study two data-related research questions:

- **RQ1 (data quantity):** How does the number of training images per class (5 / 10 / 30 / 100) affect accuracy and macro-F1?
- **RQ2 (class imbalance):** When one class (`trash`) has far fewer training images than the others (3 / 10 / 30 vs. 100), how much does its recall drop, which classes absorb its errors, and does a class-weighted loss help?

## Repository structure

```
├── environment.yml          # Conda environment (pinned versions)
├── scripts/
│   ├── download_data.sh     # Download the dataset from Kaggle
│   └── run_all.sh           # Run all experiments + aggregate results
├── src/
│   ├── common.py            # Classes, transforms, dataset, model, seeding
│   ├── make_splits.py       # Fixed test set + training pool
│   ├── train.py             # Train one condition, evaluate on the test set
│   ├── evaluate.py          # Inference/evaluation (also usable on saved weights)
│   └── aggregate.py         # Tables (CSV) and figures (PNG/PDF)
├── splits/                  # test.csv and pool.csv (exact file lists)
├── runs/                    # Per-run metrics.json and train_list.csv
└── results/                 # Aggregated tables and figures used in the paper
```

## Data

| | |
|---|---|
| Source | Kaggle, *Garbage Classification* (asdasdasasdas/garbage-classification) |
| License | Copyright of the original authors. Images are **not** redistributed here; use the download script. |
| Classes | cardboard, glass, metal, paper, plastic, trash |
| Images | 2,527 RGB photos (512×384), one object on a plain background |

Class distribution and splits (`src/make_splits.py`, seed 42):

| Class | Total | Test (fixed) | Training pool |
|---|---:|---:|---:|
| cardboard | 403 | 25 | 378 |
| glass | 501 | 25 | 476 |
| metal | 410 | 25 | 385 |
| paper | 594 | 25 | 569 |
| plastic | 482 | 25 | 457 |
| trash | 137 | 25 | 112 |

- The **test set is balanced and identical for every experiment** (150 images). It is never used for model selection.
- Each run samples its training set from the pool without replacement (seeded). The exact list is saved to `runs/<run_name>/train_list.csv`.
- The archive contains the same images twice (upper/lower-case folder names); only one copy is used.

## Experimental setup

| Setting | Value |
|---|---|
| Model | ResNet-18, ImageNet-1K pretrained (torchvision), new 6-way head, all layers fine-tuned |
| Input | Train: RandomResizedCrop 224 (scale 0.6–1.0) + horizontal flip. Test: Resize 256 → CenterCrop 224 |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4, batch size 32 |
| Training budget | **500 steps for every run** (batches sampled with replacement), so conditions differ only in data |
| Loss | Cross-entropy (CE); RQ2 also uses class-weighted CE with w_c = N / (C · N_c) |
| Seeds | 0, 1, 2 (affect training-subset sampling, initialisation of the head, and data order) |
| Metrics | Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix |

Conditions:

| Experiment | Images per class | Trash images | Loss | Runs |
|---|---|---|---|---|
| RQ1 | 5, 10, 30, 100 | = images per class | CE | 4 × 3 seeds |
| RQ2 | 100 | 30, 10, 3 | CE, weighted CE | 6 × 3 seeds |

The balanced point of RQ2 (100 trash images) reuses the RQ1 runs with 100 images per class.

## Hardware

TODO: fill in after running (GPU model from `nvidia-smi`, CUDA version, total runtime).

The code also runs on CPU (`--device cpu`), but much more slowly.

## Reproducing the results

### 1. Clone and create the environment

```bash
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>
conda env create -f environment.yml
conda activate pa04
```

### 2. Download the data

Create a Kaggle API token (Kaggle → Settings → API) and make it available to the client:

```bash
export KAGGLE_API_TOKEN=<your token>
bash scripts/download_data.sh data
```

Images will be in `data/Garbage classification/Garbage classification/<class>/`.

### 3. Run all experiments

```bash
bash scripts/run_all.sh
```

This trains and evaluates all 30 runs and writes `runs/*/metrics.json`, model weights in `checkpoints/`, and the tables/figures in `results/`. To change paths:

```bash
DATA_ROOT="/path/to/Garbage classification/Garbage classification" CKPT_DIR=/path/to/ckpt bash scripts/run_all.sh
```

A single condition can be run directly, e.g.

```bash
python src/train.py --data_root "data/Garbage classification/Garbage classification" \
    --run_name rq2_t3_ce_s0 --per_class 100 --trash 3 --seed 0
```

### 4. Evaluate the released weights (no training needed)

Download the weights from TODO (GitHub Releases link) into `checkpoints/`, then:

```bash
python src/evaluate.py --ckpt checkpoints/rq1_n100_s0.pt \
    --data_root "data/Garbage classification/Garbage classification"
```

## Expected results

TODO: copy the final numbers from `results/rq1_table.csv` and `results/rq2_table.csv` (mean ± std over 3 seeds).

Small differences (typically < 1 percentage point) can occur across GPU models and driver versions even with fixed seeds.
