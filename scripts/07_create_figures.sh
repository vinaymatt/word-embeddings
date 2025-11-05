#!/bin/bash
#
# Create Figures
# Generates all paper figures
#

set -e

echo "Creating figures..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
CONFIG="config/network_config.yaml"
NETWORKS_DIR="results/networks"
OUTPUT_DIR="results/figures"
LOG_DIR="logs"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Example: Visualize 2020 network
YEAR=2020
if [ -f "$NETWORKS_DIR/${YEAR}_network.json" ]; then
    echo "Creating network visualization for year $YEAR..."
    python src/07_visualization/network_plots.py \
        --config "$CONFIG" \
        --network "$NETWORKS_DIR/${YEAR}_network.json" \
        --output "$OUTPUT_DIR/network_${YEAR}.html" \
        --title "Similarity Network $YEAR" \
        2>&1 | tee "$LOG_DIR/07_create_figures.log"
fi

echo "✓ Figure generation complete"
echo "  Output: $OUTPUT_DIR/"
echo "  Log: $LOG_DIR/07_create_figures.log"
echo ""
echo "Note: For full paper figure generation, see notebooks/07_figure_generation.ipynb"

