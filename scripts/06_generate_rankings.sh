#!/bin/bash
#
# Generate Protein Rankings
# Ranks proteins from similarity networks
#

set -e

echo "Generating protein rankings..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
CONFIG="config/network_config.yaml"
EMBEDDINGS_DIR="data/embeddings"
OUTPUT_FILE="results/tables/protein_rankings.csv"
LOG_DIR="logs"

# Create output directories
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$LOG_DIR"

# Generate rankings
python src/06_ranking/protein_ranking.py \
    --config "$CONFIG" \
    --embeddings "$EMBEDDINGS_DIR" \
    --output "$OUTPUT_FILE" \
    --start-year 2000 \
    --end-year 2023 \
    2>&1 | tee "$LOG_DIR/06_generate_rankings.log"

echo "✓ Protein ranking complete"
echo "  Output: $OUTPUT_FILE"
echo "  Log: $LOG_DIR/06_generate_rankings.log"

