# Stack Overflow Question Quality Predictor

An end-to-end machine learning pipeline that predicts Stack Overflow question quality — from raw XML data ingestion through feature engineering, clustering, interactive labeling, and model training.

Built as a learning project to explore the full ML lifecycle using real-world data from the [Stack Exchange Data Dump](https://archive.org/details/stackexchange).

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────┐
│  1. Ingestion    │────▶│  2. Feature Pipeline  │────▶│  3. Clustering     │
│  (XML → CSV)     │     │  (Text + Structured)  │     │  (UMAP + Bokeh)    │
└─────────────────┘     └──────────────────────┘     └───────────────────┘
                                                              │
                        ┌──────────────────────┐     ┌────────▼──────────┐
                        │  5. Model Training    │◀───│  4. Labeling       │
                        │  (Random Forest / LR) │     │  (Streamlit App)   │
                        └──────────────────────┘     └───────────────────┘
```

### Stage Details

| Stage | Module | Description |
|-------|--------|-------------|
| **1. Ingestion** | `src/ingestion.py` | Parses Stack Exchange XML, cleans HTML from post bodies, creates quality labels, and performs group-aware train/test split (no author leakage) |
| **2. Feature Engineering** | `src/pipeline_orchestrator.py` | Orchestrates a scikit-learn `Pipeline` combining SpaCy text vectorization (GloVe), numeric/categorical normalization, and custom quality features (readability, formatting effort) |
| **3. Clustering** | `src/clustering_data.py` | Applies UMAP dimensionality reduction on the feature matrix for 2D visualization |
| **4. Labeling** | `src/labeler_app.py` | Streamlit interactive app for manually labeling data points on the UMAP scatter plot |
| **5. Training** | `src/model_training.py` | Trains a classifier (Random Forest or Logistic Regression), evaluates with precision/recall/F1, and saves model artifacts |

### Key Components

| Module | Role |
|--------|------|
| `src/text_vectorizer.py` | Custom sklearn transformer that converts text to 300-dim vectors using SpaCy's tokenizer + GloVe word vectors (10-50x faster than full NLP pipeline) |
| `src/feature_engineering.py` | Custom sklearn transformer extracting quality signals: title/body length, question marks, caps ratio, Flesch readability, action verb density |
| `src/numeric_categorical_vectorizer.py` | Factory for structured data preprocessing (imputation, scaling, one-hot encoding, date feature extraction) |
| `src/text_analyzer.py` | NLP utilities: syllable counting, Flesch Reading Ease scoring, writing suggestions |
| `src/visualizer.py` | Bokeh-based interactive HTML visualizations for cluster exploration and labeling |
| `src/utils.py` | Helper functions for Q&A matching, data export, and score distribution plots |

## Setup

### Prerequisites

- Python 3.9+
- A Stack Exchange data dump XML file (e.g., `Posts.xml` from [Writing Stack Exchange](https://archive.org/details/stackexchange))

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/basic_ML_pipeline.git
cd basic_ML_pipeline

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the SpaCy language model (required for text vectorization)
python -m spacy download en_core_web_md
```

### Data Setup

Place your Stack Exchange XML dump at `data/raw/Posts.xml`. You can download one from the [Stack Exchange Data Dump on Archive.org](https://archive.org/details/stackexchange) — the Writing Stack Exchange dump is a good starting point since it's smaller than Stack Overflow.

## Usage

Run the interactive CLI menu:

```bash
python main.py
```

Options:

| # | Action | What it does |
|---|--------|-------------|
| 1 | **Ingest and Clean Data** | Parses XML → cleans HTML → splits into train/test CSVs |
| 2 | **Run Pipeline Orchestrator** | Fits the full feature engineering pipeline (vectorization + normalization) |
| 3 | **Cluster and Visualize** | Runs UMAP and generates an interactive Bokeh HTML explorer |
| 4 | **Generate Sample Data** | Creates a 1,000-row sample CSV for quick testing |
| 5 | **Launch Labeling Tool** | Opens the Streamlit app for interactive data labeling |
| 6 | **Train and Evaluate Model** | Trains a classifier and prints performance metrics |

### Typical Workflow

```bash
# Step 1: Ingest raw data
python main.py  # Choose option 1

# Step 2: Explore clusters (optional)
python main.py  # Choose option 3

# Step 3: Train the model
python main.py  # Choose option 6
```

## Configuration

All parameters are centralized in `config.yaml`:

```yaml
# Key configuration sections:
paths:          # Input/output file paths
parsing:        # XML parsing settings (limit rows for testing)
splitting:      # Train/test split ratio, random seed, group column
features:       # Which features to include (numeric, categorical, text)
text_vectorizer: # SpaCy model selection and SVD compression settings
model:          # Algorithm choice, hyperparameters (Random Forest, Logistic Regression)
```

## Project Structure

```
basic_ML_pipeline/
├── main.py                  # CLI entry point
├── config.yaml              # Centralized configuration
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                 # Raw XML data (gitignored)
│   ├── processed/           # Train/test CSVs (gitignored)
│   └── html_output/         # Interactive visualizations
├── models/                  # Saved model artifacts
│   └── metadata.json        # Training metadata
├── src/
│   ├── ingestion.py         # Data loading, cleaning, splitting
│   ├── pipeline_orchestrator.py  # Full sklearn pipeline assembly
│   ├── text_vectorizer.py   # SpaCy GloVe text vectorization
│   ├── feature_engineering.py    # Custom quality feature extraction
│   ├── numeric_categorical_vectorizer.py  # Structured data preprocessing
│   ├── clustering_data.py   # UMAP dimensionality reduction
│   ├── model_training.py    # Model training and evaluation
│   ├── labeler_app.py       # Streamlit interactive labeling tool
│   ├── text_analyzer.py     # NLP text analysis utilities
│   ├── visualizer.py        # Bokeh interactive visualizations
│   └── utils.py             # Helper functions
└── tests/
    └── test_feature_engineering.py  # Unit tests
```

## Key Learnings & Design Decisions

- **Group-aware splitting**: Used `GroupShuffleSplit` to split by author (`OwnerUserId`), preventing data leakage where the model could learn author-specific patterns instead of question quality
- **Tokenizer-only vectorization**: Instead of running SpaCy's full NLP pipeline (with the expensive tok2vec CNN), we use only the tokenizer + vocabulary lookup for ~10-50x speedup with identical results
- **Quality-based labeling**: Rather than using a simple score threshold, questions are labeled using multiple signals (closed status, accepted answers, score) for more meaningful quality categories
- **Custom sklearn transformers**: Built `TextVectorizer` and `QualityFeatureExtractor` as proper sklearn transformers so they integrate seamlessly with `Pipeline` and `ColumnTransformer`

## License

This project is for educational purposes. The Stack Exchange data is used under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
