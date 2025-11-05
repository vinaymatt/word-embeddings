# Word Embeddings for Explainable Hypothesis Generation in Biomechanics and Mechanobiology

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the complete, reproducible pipeline for our paper:  
**"Word Embeddings for Explainable Hypothesis Generation in Biomechanics and Mechanobiology"**

## Overview

This work demonstrates that lightweight, interpretable word-embedding methods (Skip-Gram) can recover latent biomechanical knowledge without manual labeling. Starting from ~9 million PubMed abstracts, we:

1. Built a classifier to filter relevant papers
2. Trained yearly Skip-Gram models tracking mechanobiology concept emergence
3. Constructed weighted similarity networks for ranking candidate proteins
4. Generated biologically plausible hypotheses with sentence-level traceability

**Key advantage over LLMs:** Every prediction can be traced back to specific sentences in the source corpus.

## Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Step-by-Step Reproduction](#-step-by-step-reproduction)
- [Pipeline Architecture](#-pipeline-architecture)
- [Configuration](#-configuration)
- [Case Study: Endothelial NO Production](#-case-study-endothelial-no-production)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

## Key Features

- **Interpretable embeddings**: Skip-Gram with cosine similarity (no black-box models)
- **Year-wise tracking**: Cumulative models show concept evolution (1929-2023)
- **Z-score normalization**: Fair cross-year comparison of similarity scores
- **Weighted similarity networks**: Edge width/darkness ∝ semantic proximity
- **Sentence-level auditing**: Every hypothesis traces to PubMed contexts
- **Computationally efficient**: Re-trainable on standard hardware

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/vinaymatt/word-embeddings.git
cd word-embeddings/

# Create conda environment
conda env create -f environment.yml
conda activate NLPprocessing

# Download Spacy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_sm
```

## Quick Start

### Run Full Pipeline

```bash
# Activate environment
conda activate NLPprocessing

# Run complete pipeline
bash scripts/run_pipeline.sh
```

## Step-by-Step Reproduction

### 1. PubMed Data Collection

**Retrieve abstracts** from PubMed (1929-2023):

```bash
python src/01_pubmed_retrieval/retrieve_abstracts.py \
    --config config/pubmed_query.yaml \
    --output data/raw/ \
    --start-year 1929 \
    --end-year 2023
```

**Expected output:** ~9 million abstracts in `data/raw/YYYY.txt`  
**Query:** `{protein OR {endothelial cell}} OR {shear stress OR {nitric oxide}}`

### 2. Preprocessing

**Filter English abstracts:**

```bash
python src/02_preprocessing/language_filter.py \
    --input data/raw/ \
    --output data/english/
```


**Apply preprocessing pipeline:**

```bash
python src/02_preprocessing/preprocess.py \
    --config config/preprocessing_config.yaml \
    --input data/english/ \
    --output data/processed/
```

**Preprocessing steps:**
- Metadata tag removal (BACKGROUND, AIMS, METHODS, etc.)
- Lemmatization (Spacy en_core_web_sm)
- NER-based entity preservation (proteins, genes)
- Number tokenization (`<num>`)
- Selective lowercasing
- Bigram/trigram detection (Gensim Phrases)

### 3. Classification

The classification pipeline has **three steps**:

#### 3a. GPT-4 Labeling (Bootstrapping)

**Generate initial labeled dataset using GPT-4:**

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Label abstracts with GPT-4
python src/03_classification/gpt4_labeling.py \
    --api-key $OPENAI_API_KEY \
    --abstracts data/sample_abstracts.txt \
    --output data/classifier_training/labeled_abstracts.xlsx \
    --model gpt-4 \
    --prompt-version refined \
    --resume \
    --organize \
    --organize-output data/classifier_training/
```


**Key Features:**
- Model: GPT-4 (Please change as newer models emerge)
- Temperature: 0 (deterministic)
- Retry logic: 5 attempts on API errors
- Incremental saving: Prevents data loss
- Outputs: Excel file with 'Abstract' and 'Relevance' columns

#### 3b. Train MNB Classifier

**Train Multinomial Naive Bayes on GPT-4 labeled data:**

```bash
python src/03_classification/train_mnb.py \
    --config config/classifier_config.yaml \
    --data data/classifier_training/labeled_abstracts.xlsx \
    --output models/
```

**Classifier Details:**
- **Model:** `sklearn.naive_bayes.MultinomialNB()` (default parameters)
- **Features:** Bag-of-words with `CountVectorizer`
  - `analyzer='word'`
  - `token_pattern=r'\w{1,}'`
  - No binary encoding (standard counts)
- **Training:** 50 expert-labeled abstracts (validated by domain expert)
- **Validation:** 80/20 split, stratified, `random_state=42`

**Evaluation Metrics:**
- Precision, Recall, F1-score
- Confusion Matrix (TN, FP, FN, TP)
- ROC-AUC
- Cohen's Kappa

**Why MNB over GPT-4 for production?**
1. **High precision** at operating point (minimizes false positives)
2. **Transparent term-level weights** (interpretable via `log_prob`)
3. **Linear-time inference** (scalable to 9M abstracts)
4. **No API costs** (GPT-4 would cost $$$$ for 9M abstracts)
5. **Deterministic** (same results every time)

#### 3c. Apply Classifier to Full Corpus

**Filter the 9M abstracts using trained MNB:**

```bash
python src/03_classification/apply_classifier.py \
    --config config/classifier_config.yaml \
    --model models/ \
    --input data/processed/ \
    --output data/classified/ \
    --min-prob 0.5
```

**Process:**
- Loads trained MNB model
- Processes preprocessed abstracts year-by-year
- Retains only abstracts classified as "RELEVANT" with probability ≥ 0.5
- Saves to `data/classified/YYYY_abstracts_classified.txt`

**Or run all three steps together:**

```bash
bash scripts/03_train_classifier.sh
```

### 4. Embedding Training

**Train Skip-Gram embeddings (year-wise cumulative):**

```bash
python src/04_embeddings/train_skipgram.py \
    --config config/embedding_config.yaml \
    --input data/classified/ \
    --output data/embeddings/ \
    --seed 42
```

**Hyperparameters (from config/embedding_config.yaml):**
- Embedding size: 200
- Window size: 8
- Min count: 5
- Negative sampling: 10
- Epochs: 30
- Learning rate: 0.001 (fixed)
- Decay rate: 0.001

**Training strategy:**
- **Cumulative:** Year N includes all abstracts from 1929 to N
- **Phrase detection:** Bigrams/trigrams merged (e.g., `nitric_oxide`, `cell_membrane`)
- **Random seed:** 42 (for reproducibility)

**Output:** 95 embedding files (1929-2023), 200D vectors

### 5. Graph Construction

**Build weighted similarity networks:**

```bash
python src/05_graph_construction/similarity_network.py \
    --config config/network_config.yaml \
    --embeddings data/embeddings/ \
    --output results/networks/
```

**Network construction:**
1. Compute pairwise cosine similarity matrix
2. Z-score normalization (mean + 2.5 std threshold)
3. Build weighted NetworkX graph
4. Save as JSON (node-link format)

**Why z-score normalization?**
- Enables fair cross-year comparison
- Similarities follow Gaussian distribution (see Supplementary Fig S1)

**Output:** JSON network files per year

### 6. Protein Ranking

**Rank proteins for endothelial NO case study:**

```bash
python src/06_ranking/protein_ranking.py \
    --config config/network_config.yaml \
    --embeddings data/embeddings/ \
    --output results/tables/protein_rankings.csv \
    --start-year 2000 \
    --end-year 2023
```

**Input words (from config):**
- dilation
- platelet
- aggregation
- atherosclerosis
- nitric_oxide
- vasodilation
- endothelium-dependent_vasorelaxation
- shear_stress-induced
- NO-mediated_vasodilation

**NER filtering:**
- SpaCy en_core_sci_sm
- Entity types: PROTEIN, GENE, CHEMICAL

**Output:** CSV with year-by-year protein rankings

### 7. Visualization

**Create interactive network plots:**

```bash
python src/07_visualization/network_plots.py \
    --config config/network_config.yaml \
    --network results/networks/2020_network.json \
    --output results/figures/network_2020.html \
    --title "Similarity Network 2020"
```

**Visualization features:**
- Interactive Plotly plots
- Node size ∝ degree
- Edge width ∝ similarity
- Spring layout (seed=42, reproducible)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PubMed Retrieval                            │
│  Query: {protein OR endothelial cell} OR {shear stress OR NO}   │
│  → ~9 million abstracts (1929-2023)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Preprocessing                               │
│  • English filtering (langdetect)                               │
│  • Lemmatization (Spacy en_core_web_sm)                         │
│  • NER entity preservation (en_ner_jnlpba, en_ner_bionlp)       │
│  • Bigram/trigram detection (Gensim Phrases)                    │
│  • Number tokenization (<num>)                                  │
│  • Metadata tag removal (BACKGROUND, METHODS, etc.)             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            GPT-4 Labeling (Bootstrapping)                       │
│  • Sample ~1,000 abstracts for labeling                         │
│  • GPT-4 API (temperature=0, max_tokens=10)                     │
│  • Prompt: "RELEVANT or NOT RELEVANT"                           │
│  • Expert validation of results                                 │
│  → Creates: labeled_abstracts.xlsx (50 expert-validated).       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Train MNB Classifier (Production)                       │
│  • Train on 50 GPT-4 labeled abstracts                          │
│  • Features: Bag-of-words (CountVectorizer)                     │
│  • Model: MultinomialNB (sklearn, default params)               │
│  • Evaluation: Precision, Recall, F1, Cohen's Kappa, ROC-AUC    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Apply MNB to Full Corpus (9M abstracts)                 │
│  • Batch processing (1000 abstracts/batch)                      │
│  • Filter by probability threshold (≥0.5)                       │
│  • Year-by-year processing                                      │
│  → Retains ~30-50% of corpus                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Embedding Training (Skip-Gram)                        │
│  • Year-wise cumulative training (1929-2023)                    │
│  • 200D vectors, window=8, min_count=5                          │
│  • Negative sampling=10, epochs=30                              │
│  • seed=42 (reproducibility)                                    │
│  • Phrase detection: min_count=7, threshold=15                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Graph Construction + Z-score Normalization              │
│  • Cosine similarity matrix (pairwise)                          │
│  • Z-score normalization (mean + 2.5 std threshold)             │
│  • Weighted NetworkX graphs (JSON export)                       │
│  • Enables cross-year comparison                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Protein Ranking + NER Filtering                        │
│  • Intersection network (10 input words)                        │
│  • SpaCy en_core_sci_sm (biomedical NER)                        │
│  • Top-20 proteins per year (2000-2023)                         │
│  • Year-by-year ranking evolution                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Results                                    │
│  • Figures: Interactive Plotly HTML networks                    │
│  • Tables: protein_rankings.csv (year-by-year)                  │
│  • Networks: JSON files (1929-2023)                             │
│  • All results auditable to source PubMed sentences             │
└─────────────────────────────────────────────────────────────────┘

```


## Contact

- **Vinay Saji Mathew**: [vvm5242@psu.edu]



## Acknowledgments

- OpenAI for classifier bootstrapping
- PubMed/NCBI for biomedical literature access
- Tshitoyan et al. (2019) for methodological inspiration

---


