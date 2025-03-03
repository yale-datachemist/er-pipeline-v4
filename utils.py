#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility module for the entity resolution pipeline.
Provides shared functionality across modules.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up the logger for the pipeline.
    
    Args:
        log_level: Logging level (default: "INFO")
        log_file: Path to log file (optional)
    
    Returns:
        Configured logger
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create and configure file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        logging.info("Using default configuration")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the pipeline.
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Directories
        "input_dir": "data/input",
        "output_dir": "data/output",
        "checkpoint_dir": "data/checkpoints",
        "log_dir": "logs",
        
        # Preprocessing settings
        "save_checkpoints": True,
        
        # Development mode settings
        "dev_mode": False,
        "dev_max_files": 5,
        "dev_max_rows": 1000,
        
        # Embedding settings
        "embedding_model": "text-embedding-3-small",
        "embedding_batch_size": 100,
        "embedding_chunk_size": 8192,
        "embedding_dimension": 1536,
        
        # Weaviate settings
        "weaviate_url": "http://localhost:8080",
        "weaviate_batch_size": 100,
        "weaviate_timeout": 300,
        
        # Classification settings
        "train_test_split": 0.8,
        "random_seed": 42,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "regularization": 0.01,
        "feature_normalization": True,
        "confidence_threshold": 0.7,
        "candidate_limit": 100,
        "classification_batch_size": 1000,
        
        # LLM fallback settings
        "use_llm_fallback": False,
        "llm_model": "gpt-4o",
        "max_llm_requests": 1000,
        
        # Logging settings
        "log_level": "INFO"
    }


def extract_years(text: str) -> Set[int]:
    """
    Extract years from a text string.
    
    Args:
        text: Input text
    
    Returns:
        Set of extracted years as integers
    """
    if not text:
        return set()
    
    # Find 4-digit years between 1400 and current year + 5
    year_pattern = r'\b(1[4-9]\d\d|20\d\d)\b'
    years = set(int(year) for year in re.findall(year_pattern, text))
    return years


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity (between -1 and 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def normalize_features(features: List[float]) -> List[float]:
    """
    Normalize a feature vector to have values between 0 and 1.
    
    Args:
        features: Feature vector
    
    Returns:
        Normalized feature vector
    """
    # Convert to numpy array for easier operations
    features_array = np.array(features)
    
    # Check for zero range
    min_val = np.min(features_array)
    max_val = np.max(features_array)
    
    if max_val == min_val:
        return features  # No need to normalize if all values are the same
    
    # Normalize to [0, 1]
    normalized = (features_array - min_val) / (max_val - min_val)
    
    return normalized.tolist()


def save_output(file_path: str, data: Any, config: Dict[str, Any]) -> None:
    """
    Save output data to a file.
    
    Args:
        file_path: Output file path
        data: Data to be saved
        config: Configuration dictionary
    """
    output_dir = config.get("output_dir", "data/output")
    full_path = os.path.join(output_dir, file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Determine file format and save accordingly
    if file_path.endswith('.json'):
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif file_path.endswith('.jsonl'):
        with open(full_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif file_path.endswith('.csv'):
        import csv
        with open(full_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        raise ValueError(f"Unsupported output file format: {file_path}")
    
    logging.info(f"Saved output to {full_path}")


def batch_generator(items: List[Any], batch_size: int):
    """
    Generate batches from a list of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Utility module loaded successfully")
