#!/usr/bin/env bash
# Download the Garbage Classification dataset from Kaggle.
# Requires a Kaggle API token (see README). Usage:
#   bash scripts/download_data.sh [OUTPUT_DIR]     (default: data)
set -e
OUT="${1:-data}"
kaggle datasets download -d asdasdasasdas/garbage-classification -p "$OUT" --unzip
# The archive contains the same images twice (upper/lower-case folder names).
rm -rf "$OUT/garbage classification"
echo "Images are in: $OUT/Garbage classification/Garbage classification"
