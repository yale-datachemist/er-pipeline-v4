## Enhanced Feature Engineering

The system includes advanced feature engineering capabilities that significantly improve matching performance:

### Enhanced Temporal Analysis

The basic temporal overlap feature is extended with detailed analysis:

- **Temporal Span Similarity**: Compares the publication timespan of each author
- **Publication Era Match**: Checks if works were published in the same historical eras
- **Year Difference Metrics**: Analyzes minimum, maximum, and median year differences
- **Publication Density Similarity**: Compares how concentrated publications are across time
- **Temporal Sequence Matching**: Analyzes the chronological pattern of publications
- **Life Dates Integration**: Compares life dates (birth/death years) to publication years
- **Biographical Plausibility**: Assesses if publication timelines make sense given life dates
- **Posthumous Publication Detection**: Identifies and properly handles works published long after an author's death
- **Historical Publication Patterns**: Recognizes patterns common for classical/historical authors (e.g., Shakespeare, Plato)
- **Long-term Publishing Analysis**: Handles publication timeframes spanning centuries

These features include robust handling of data quality issues:

- **Fuzzy Year Matching**: Detects and accommodates off-by-a-few-years errors
- **Transposed Digit Detection**: Identifies years with swapped digits (e.g., 1879 vs 1897)
- **Century Error Handling**: Recognizes when years differ only by century (e.g., 1792 vs 1892)
- **Single-Digit Errors**: Accounts for typographical errors in individual digits

These features help distinguish between different authors with the same name who published in different eras or with different publication patterns.

### Feature Interaction Terms

The system generates interaction terms between important features to capture non-linear relationships:

- **Semantic-String Interactions**: Combines semantic vector similarity with string similarity
- **Temporal-Person Interactions**: Captures the relationship between name similarity and temporal patterns
- **Domain-Temporal Interactions**: Links subject/genre patterns with publication timeframes
- **Strong Signal Boosting**: Amplifies the signal when multiple strong indicators align
- **Non-linear Transformations**: Applies mathematical transformations to emphasize high-confidence matches

Feature interactions are particularly valuable for handling complex cases like posthumous publications or authors who published across different domains.

### Using Enhanced Features

Enhanced features are enabled by default. You can control them in the configuration:

```json
{
  "use_enhanced_features": true,
  "enhanced_temporal_features": true,
  "enhanced_feature_interactions": true
}
```

To disable them for faster processing with simpler models:

```json
{
  "use_enhanced_features": false
}
```# Entity Resolution for Library Catalog Entities

This project implements an entity resolution system for the Yale University Library catalog. It resolves personal name entities across catalog records using vector embeddings, machine learning classification, and graph-based clustering.

## Overview

The system performs entity resolution through the following steps:

1. **Preprocessing**: Parse CSV files, normalize strings, and deduplicate values
2. **Vector Embedding**: Generate 1,536-dimensional embeddings for unique strings using OpenAI's models
3. **Weaviate Indexing**: Index embeddings for efficient similarity search
4. **Feature Engineering**: Create feature vectors for record pairs
5. **Classification**: Train a logistic regression classifier to predict matches
6. **Clustering**: Apply graph-based clustering to form entity groups
7. **Analysis**: Generate detailed reports and visualizations for each pipeline stage

## Project Structure

```
entity-resolution/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker setup for Weaviate
├── config.json              # Configuration settings
├── pipeline.py              # Main pipeline module
├── preprocessing.py         # Data preprocessing module
├── embedding.py             # Vector embedding module
├── indexing.py              # Weaviate integration module
├── classification.py        # Feature engineering and classification module
├── analysis.py              # Analysis and reporting module
├── utils.py                 # Utility functions
├── data/                    # Data directory
│   ├── input/               # Input CSV files
│   ├── output/              # Output results
│   └── checkpoints/         # Saved checkpoints
└── logs/                    # Log files
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/entity-resolution.git
   cd entity-resolution
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Weaviate using Docker:
   ```
   docker-compose up -d
   ```

4. Set the OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Configuration

Edit `config.json` to customize the pipeline settings. Key configuration options include:

- `input_dir`: Directory containing input CSV files
- `dev_mode`: Run with a subset of data for development
- `embedding_model`: OpenAI embedding model to use
- `confidence_threshold`: Threshold for entity matching
- `use_llm_fallback`: Whether to use LLM for ambiguous cases
- `use_enhanced_features`: Enable advanced feature engineering
- `enhanced_temporal_features`: Enable detailed temporal analysis
- `enhanced_feature_interactions`: Enable feature interaction terms
- `robust_date_handling`: Enable robust handling of date errors and inconsistencies
- `fuzzy_year_threshold`: Configure tolerance level for year matching (default: 5)

### Running the Full Pipeline

To run the complete pipeline:

```
python pipeline.py
```

For development mode with a subset of data:

```
python pipeline.py --dev
```

To run the pipeline without generating analysis reports:

```
python pipeline.py --no-analysis
```

### Running Individual Modules

You can run specific modules independently:

```
python pipeline.py --module preprocessing
python pipeline.py --module embedding
python pipeline.py --module indexing
python pipeline.py --module feature_engineering
python pipeline.py --module training
python pipeline.py --module classification
python pipeline.py --module evaluation
python pipeline.py --module analysis
```

## Pipeline Steps

### 1. Preprocessing

The preprocessing module parses CSV files and extracts field values. It deduplicates strings to optimize embedding generation and builds the following data structures:

- `unique_strings`: Hash → String value mapping
- `string_counts`: Frequency counts for each unique string
- `record_field_hashes`: Record ID → Field → Hash mapping
- `field_hash_mapping`: Hash → Field → Count mapping

### 2. Vector Embedding

The embedding module generates 1,536-dimensional vector embeddings for each unique string using OpenAI's `text-embedding-3-small` model. It handles:

- Batch processing with configurable batch size
- Rate limiting to respect API quotas
- Checkpoint saving for fault tolerance

### 3. Weaviate Indexing

The indexing module stores embeddings in Weaviate for efficient similarity search:

- Creates a collection with named vectors for each field type
- Indexes strings with their corresponding field types
- Configures HNSW parameters for optimal search performance

### 4. Feature Engineering & Classification

The classification module:

- Generates feature vectors for record pairs
- Imputes missing values using vector-based hot deck approach
- Trains a logistic regression classifier with gradient descent
- Implements optional LLM fallback for ambiguous cases
- Provides enhanced feature engineering with temporal analysis and feature interactions

### 5. Clustering

The clustering module:

- Builds a graph with records as nodes and matches as edges
- Applies community detection to identify entity clusters
- Derives canonical names for each cluster
- Calculates confidence scores for clusters

## Evaluation

The system evaluates entity resolution performance using:

- Precision: Percentage of predicted matches that are correct
- Recall: Percentage of actual matches that are predicted
- F1 Score: Harmonic mean of precision and recall

## Analysis and Reporting

The system generates comprehensive analysis reports and visualizations for each pipeline stage:

### Preprocessing Analysis
- Field statistics and distributions
- String frequency analysis
- Person name length distribution

### Embedding Analysis
- Embeddings distribution via PCA and t-SNE
- Vector statistics by field type

### Indexing Analysis
- Indexed object counts by field type
- Weaviate performance metrics

### Feature Engineering Analysis
- Feature importance scores
- Feature correlation heatmaps
- Feature distributions by class

### Classification Analysis
- Confusion matrix visualization
- ROC and precision-recall curves
- Decision boundary analysis

### Clustering Analysis
- Cluster size distribution
- Confidence value distribution
- Size vs. confidence correlations

### Pipeline Analysis
- Stage timing breakdown
- Resource usage statistics
- Overall performance metrics

Reports and visualizations are saved to the `data/output/reports` directory.

## Resources

- Vector Embedding: Uses OpenAI's `text-embedding-3-small` model
- Vector Search: Uses Weaviate (minimum version 1.24.x)
- Classification: Custom logistic regression with gradient descent
- Clustering: Graph-based community detection with NetworkX

## License

[Include your license information here]
