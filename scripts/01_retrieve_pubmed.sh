#!/bin/bash
#
# Retrieve PubMed Abstracts
# Fetches abstracts from PubMed using Entrez API
#

set -e

echo "Retrieving PubMed abstracts..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
CONFIG="config/pubmed_query.yaml"
OUTPUT_DIR="data/raw"
LOG_DIR="logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Check if email is set in config
EMAIL=$(grep '^email:' "$CONFIG" | cut -d':' -f2 | tr -d ' "' | tr -d "'")

if [ "$EMAIL" = "your.email@example.com" ]; then
    echo "⚠️  WARNING: Please set your email in $CONFIG"
    echo "  NCBI requires a valid email for API access."
    echo ""
    read -p "Enter your email address: " USER_EMAIL
    
    # Update config
    sed -i.bak "s/your.email@example.com/$USER_EMAIL/" "$CONFIG"
    echo "✓ Updated email in $CONFIG"
fi

# Retrieve abstracts
python src/01_pubmed_retrieval/retrieve_abstracts.py \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/01_retrieve_pubmed.log"

echo "✓ PubMed retrieval complete"
echo "  Output: $OUTPUT_DIR/"
echo "  Log: $LOG_DIR/01_retrieve_pubmed.log"

