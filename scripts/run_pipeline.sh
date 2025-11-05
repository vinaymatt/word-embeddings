#!/bin/bash
#
# Master Pipeline Script
# Runs the complete PNAS Biomechanics Hypothesis Generation Pipeline
#
# Usage: bash scripts/run_pipeline.sh [--skip-retrieval] [--skip-preprocessing]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo "================================================================"
echo "  PNAS Biomechanics Hypothesis Generation Pipeline"
echo "  Word Embeddings for Explainable Hypothesis Generation"
echo "================================================================"
echo ""

# Parse command-line arguments
SKIP_RETRIEVAL=false
SKIP_PREPROCESSING=false
SKIP_CLASSIFICATION=false
SKIP_EMBEDDINGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-retrieval)
            SKIP_RETRIEVAL=true
            shift
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip-classification)
            SKIP_CLASSIFICATION=true
            shift
            ;;
        --skip-embeddings)
            SKIP_EMBEDDINGS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Working directory: $BASE_DIR"
echo ""

# Step 1: PubMed Retrieval
if [ "$SKIP_RETRIEVAL" = false ]; then
    echo -e "${GREEN}Step 1/7: Retrieving PubMed abstracts...${NC}"
    bash scripts/01_retrieve_pubmed.sh
    echo ""
else
    echo -e "${YELLOW}Step 1/7: Skipping PubMed retrieval${NC}"
    echo ""
fi

# Step 2: Preprocessing
if [ "$SKIP_PREPROCESSING" = false ]; then
    echo -e "${GREEN}Step 2/7: Preprocessing abstracts...${NC}"
    bash scripts/02_preprocess.sh
    echo ""
else
    echo -e "${YELLOW}Step 2/7: Skipping preprocessing${NC}"
    echo ""
fi

# Step 3: Classification
if [ "$SKIP_CLASSIFICATION" = false ]; then
    echo -e "${GREEN}Step 3/7: Training and applying classifier...${NC}"
    bash scripts/03_train_classifier.sh
    echo ""
else
    echo -e "${YELLOW}Step 3/7: Skipping classification${NC}"
    echo ""
fi

# Step 4: Embedding Training
if [ "$SKIP_EMBEDDINGS" = false ]; then
    echo -e "${GREEN}Step 4/7: Training Skip-Gram embeddings...${NC}"
    bash scripts/04_train_embeddings.sh
    echo ""
else
    echo -e "${YELLOW}Step 4/7: Skipping embedding training${NC}"
    echo ""
fi

# Step 5: Graph Construction
echo -e "${GREEN}Step 5/7: Constructing similarity networks...${NC}"
bash scripts/05_construct_graphs.sh
echo ""

# Step 6: Protein Ranking
echo -e "${GREEN}Step 6/7: Generating protein rankings...${NC}"
bash scripts/06_generate_rankings.sh
echo ""

# Step 7: Figure Generation
echo -e "${GREEN}Step 7/7: Creating figures...${NC}"
bash scripts/07_create_figures.sh
echo ""

# Summary
echo "================================================================"
echo -e "${GREEN}Pipeline complete!${NC}"
echo "================================================================"
echo ""
echo "Results are available in:"
echo "  - Figures: results/figures/"
echo "  - Tables: results/tables/"
echo "  - Networks: results/networks/"
echo ""
echo "To explore results interactively:"
echo "  jupyter notebook notebooks/"
echo ""

