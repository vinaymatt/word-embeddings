#!/bin/bash
#
# Construct Similarity Networks
# Builds weighted similarity networks from embeddings
#

set -e

echo "Constructing similarity networks..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
CONFIG="config/network_config.yaml"
EMBEDDINGS_DIR="data/embeddings"
OUTPUT_DIR="results/networks"
LOG_DIR="logs"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Construct networks
python src/05_graph_construction/similarity_network.py \
    --config "$CONFIG" \
    --embeddings "$EMBEDDINGS_DIR" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/05_construct_graphs.log"

echo "✓ Network construction complete"
echo "  Output: $OUTPUT_DIR/"
echo "  Log: $LOG_DIR/05_construct_graphs.log"

