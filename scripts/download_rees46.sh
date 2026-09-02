#!/usr/bin/env bash

set -euo pipefail

RAW_DIR="data/raw"
BASE_URL="https://data.rees46.com/datasets/marketplace"

mkdir -p "${RAW_DIR}"

FILES=(
  "2019-Oct.csv.gz"
  "2019-Nov.csv.gz"
  "2019-Dec.csv.gz"
  "2020-Jan.csv.gz"
  "2020-Feb.csv.gz"
  "2020-Mar.csv.gz"
  "2020-Apr.csv.gz"
)

for file in "${FILES[@]}"; do
    echo
    echo "=================================================="
    echo "Downloading ${file}"
    echo "=================================================="

    curl \
      --fail \
      --location \
      --continue-at - \
      --retry 5 \
      --retry-delay 5 \
      --output "${RAW_DIR}/${file}" \
      "${BASE_URL}/${file}"
done

echo
echo "All REES46 raw archives downloaded."
