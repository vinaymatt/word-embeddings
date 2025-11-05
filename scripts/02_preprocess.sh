#!/bin/bash
#
# Preprocess Abstracts
# 1. Filter English abstracts
# 2. Apply full preprocessing pipeline
#

set -e

echo "Preprocessing abstracts..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
PREPROCESSING_CONFIG="config/preprocessing_config.yaml"
RAW_DIR="data/raw"
ENGLISH_DIR="data/english"
PROCESSED_DIR="data/processed"
LOG_DIR="logs"

# Create directories
mkdir -p "$ENGLISH_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$LOG_DIR"

# Step 1: Filter English abstracts
echo "Step 2a: Filtering English abstracts..."
python src/02_preprocessing/language_filter.py \
    --input "$RAW_DIR" \
    --output "$ENGLISH_DIR" \
    2>&1 | tee "$LOG_DIR/02a_language_filter.log"

echo "✓ English filtering complete"

# Step 2: Preprocess
echo ""
echo "Step 2b: Preprocessing (lemmatization, NER, bigrams)..."
python src/02_preprocessing/preprocess.py \
    --config "$PREPROCESSING_CONFIG" \
    --input "$ENGLISH_DIR" \
    --output "$PROCESSED_DIR" \
    2>&1 | tee "$LOG_DIR/02b_preprocess.log"

echo "✓ Preprocessing complete"
echo "  Output: $PROCESSED_DIR/"
echo "  Logs: $LOG_DIR/02*.log"

