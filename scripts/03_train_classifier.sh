#!/bin/bash
#
# Train and Apply Classifier
# 1. (Optional) Use GPT-4 to label initial abstracts
# 2. Train MNB classifier on labeled data
# 3. Apply to full corpus
#

set -e

echo "Classification pipeline..."

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Paths
CONFIG="config/classifier_config.yaml"
PROCESSED_DIR="data/processed"
CLASSIFIED_DIR="data/classified"
TRAINING_DATA="data/classifier_training/labeled_abstracts.xlsx"
MODEL_DIR="models"
LOG_DIR="logs"

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$TRAINING_DATA")"

# Step 1: GPT-4 Labeling (OPTIONAL - only if no labeled data exists)
if [ ! -f "$TRAINING_DATA" ]; then
    echo "⚠️  No labeled training data found at: $TRAINING_DATA"
    echo ""
    echo "To create labeled data with GPT-4:"
    echo "  python src/03_classification/gpt4_labeling.py \\"
    echo "    --api-key \$OPENAI_API_KEY \\"
    echo "    --abstracts data/sample_abstracts.txt \\"
    echo "    --output $TRAINING_DATA \\"
    echo "    --model gpt-4 \\"
    echo "    --prompt-version refined \\"
    echo "    --resume \\"
    echo "    --organize \\"
    echo "    --organize-output data/classifier_training/"
    echo ""
    echo "Skipping GPT-4 labeling step..."
    echo "Please provide labeled data or run the above command manually."
    echo ""
else
    echo "✓ Using existing labeled data: $TRAINING_DATA"
fi

# Step 2: Train MNB Classifier
if [ -f "$TRAINING_DATA" ]; then
    echo ""
    echo "Training MNB classifier..."
    python src/03_classification/train_mnb.py \
        --config "$CONFIG" \
        --data "$TRAINING_DATA" \
        --output "$MODEL_DIR" \
        --test-size 0.2 \
        2>&1 | tee "$LOG_DIR/03_train_classifier.log"
    
    echo "✓ Classifier training complete"
else
    echo "⚠️  Skipping classifier training (no labeled data)"
fi

# Step 3: Apply Classifier to Full Corpus
if [ -f "$MODEL_DIR/mnb_classifier_model.pkl" ]; then
    echo ""
    echo "Applying classifier to full corpus..."
    python src/03_classification/apply_classifier.py \
        --config "$CONFIG" \
        --model "$MODEL_DIR" \
        --input "$PROCESSED_DIR" \
        --output "$CLASSIFIED_DIR" \
        --min-prob 0.5 \
        2>&1 | tee -a "$LOG_DIR/03_apply_classifier.log"
    
    echo "✓ Corpus filtering complete"
    echo "  Output: $CLASSIFIED_DIR/"
    echo "  Log: $LOG_DIR/03_apply_classifier.log"
else
    echo "⚠️  No trained classifier found. Skipping corpus filtering."
    echo "  Please train classifier first or copy existing model to $MODEL_DIR/"
fi

