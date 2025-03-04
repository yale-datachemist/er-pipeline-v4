# Entity Resolution for Library Catalog Entities

This project implements a sophisticated entity resolution system for Yale University Library catalog data. It resolves personal name entities across catalog records using vector embeddings, ANN-based blocking, machine learning classification, and graph-based clustering.

## Overview

The system performs entity resolution through the following steps:

1. **Preprocessing**: Parse CSV files, normalize strings, and deduplicate values
2. **Vector Embedding**: Generate 1,536-dimensional embeddings for unique strings using OpenAI's models
3. **Weaviate Indexing**: Index embeddings for efficient approximate nearest neighbor search
4. **ANN-Based Blocking**: Use vector similarity of person names as a blocking key to efficiently find candidate pairs
5. **Feature Engineering**: Create feature vectors for record pairs with optional interaction features
6. **Classification**: Train a logistic regression classifier to predict matches
7. **Clustering**: Apply graph-based clustering to form entity groups
8. **Analysis**: Generate detailed reports and visualizations for each pipeline stage

## Recent Updates

- **ANN-Based Blocking**: Implemented efficient blocking using person vector similarity
- **High-Value Interaction Features**: Focused interaction feature set for better matching precision
- **Configurable Feature Enhancement**: Added granular control for feature types
- **Performance Improvements**: Optimized candidate pair generation with hash-based lookups
- **Enhanced Documentation**: Improved docstrings and code comments for better maintainability

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
├── enhanced_features.py     # Enhanced feature engineering module
├── integration.py           # Integration utilities
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
   ```bash
   git clone https://github.com/yourusername/entity-resolution.git
   cd entity-resolution
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Weaviate using Docker:
   ```bash
   docker-compose up -d
   ```

4. Set the OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Configuration

Edit `config.json` to customize the pipeline settings. Key configuration options include:

| Option | Description | Default |
|--------|-------------|---------|
| `input_dir` | Directory containing input CSV files | `"data/input"` |
| `dev_mode` | Run with a subset of data for development | `false` |
| `embedding_model` | OpenAI embedding model to use | `"text-embedding-3-small"` |
| `confidence_threshold` | Threshold for entity matching | `0.7` |
| `use_enhanced_features` | Enable enhanced feature engineering | `false` |
| `interaction_features_only` | Use only interaction features | `false` |
| `use_llm_fallback` | Whether to use LLM for ambiguous cases | `false` |
| `candidate_limit` | Maximum number of candidates per record | `100` |
| `classification_batch_size` | Number of records to process in one batch | `1000` |

### Feature Engineering Configuration

To control the feature engineering process, you can use these configuration options:

```json
{
  "use_enhanced_features": false,         // Master switch for all enhanced features
  "enhanced_temporal_features": false,    // Enable enhanced temporal features
  "interaction_features_only": true,      // Use only interaction features
  "feature_normalization": true          // Normalize feature vectors
}
```

## Usage

### Running the Full Pipeline

To run the complete pipeline:

```bash
python pipeline.py
```

For development mode with a subset of data:

```bash
python pipeline.py --dev
```

To run the pipeline without generating analysis reports:

```bash
python pipeline.py --no-analysis
```

### Running Individual Modules

You can run specific modules independently:

```bash
python pipeline.py --module preprocessing
python pipeline.py --module embedding
python pipeline.py --module indexing
python pipeline.py --module feature_engineering
python pipeline.py --module training
python pipeline.py --module classification
python pipeline.py --module evaluation
python pipeline.py --module analysis
```

## Key Features

### ANN-Based Blocking

The system uses Approximate Nearest Neighbor (ANN) search to efficiently find candidate pairs for comparison:

1. Each record's `person` field is embedded as a vector
2. Weaviate's HNSW algorithm finds similar vectors
3. This acts as a blocking key, drastically reducing the number of pairwise comparisons
4. A hash-based lookup map efficiently connects person vectors to record IDs

This approach is both scalable and accurate, allowing the system to handle hundreds of thousands of records efficiently.

### High-Value Interaction Features

The system includes carefully selected interaction features that capture non-linear relationships between base features:

1. **Name-Temporal Interaction**: Combines person name similarity with publication timeline compatibility
2. **String Reinforcement**: Strengthens signals when multiple string similarity measures agree
3. **Exact Match Detection**: Binary indicators for high-confidence exact matches
4. **Life Dates Integration**: Special handling for records with birth/death dates
5. **Role-Temporal Interactions**: Captures relationships between author roles and publishing patterns

These interaction features improve matching precision for bibliographic data, particularly for:
- Historical authors with publications spanning centuries
- Name variations across different records
- Posthumous publications
- Records with asymmetric information (e.g., one has life dates, one doesn't)

## Pipeline Steps

### 1. Preprocessing

The preprocessing module parses CSV files and extracts field values. It deduplicates strings to optimize embedding generation and builds the following data structures:

- `unique_strings`: Hash → String value mapping
- `string_counts`: Frequency counts for each unique string
- `record_field_hashes`: Record ID → Field → Hash mapping
- `field_hash_mapping`: Hash → Field → Count mapping

### 2. Vector Embedding

The embedding module generates 1,536-dimensional vector embeddings for each unique string using OpenAI's model. It handles:

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

- Uses ANN-based blocking to efficiently identify candidate pairs
- Generates feature vectors for record pairs
- Imputes missing values using vector-based hot deck approach
- Adds high-value interaction features
- Trains a logistic regression classifier with gradient descent
- Implements optional LLM fallback for ambiguous cases

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

## Troubleshooting

### Vector Embedding Issues

If you encounter errors during the embedding process:
- Check your OpenAI API key is correctly set
- Verify API rate limits haven't been exceeded
- Ensure the embedding model specified in config exists

### Weaviate Connection Issues

If Weaviate fails to connect:
- Verify Docker is running and the container is active
- Check the Weaviate URL in the configuration
- Ensure the port is not being used by another service

### Memory Issues

For large datasets:
- Reduce batch sizes in the configuration
- Enable development mode for testing
- Increase Docker memory allocation for Weaviate
- Consider using a machine with more RAM

## License

[Include your license information here]