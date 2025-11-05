#!/bin/bash
#
# Train Skip-Gram Embeddings
# Trains year-wise cumulative embeddings on classified abstracts
#

set -e

echo "Training Skip-Gram embeddings..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
CONFIG="config/embedding_config.yaml"
INPUT_DIR="data/classified"
OUTPUT_DIR="data/embeddings"
LOG_DIR="logs"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Train embeddings
python src/04_embeddings/train_skipgram.py \
    --config "$CONFIG" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/04_train_embeddings.log"

echo "✓ Embedding training complete"
echo "  Output: $OUTPUT_DIR/"
echo "  Log: $LOG_DIR/04_train_embeddings.log"

