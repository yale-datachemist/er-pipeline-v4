#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing module for the entity resolution pipeline.
Handles CSV parsing, string normalization, and deduplication.
"""

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import pandas as pd
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


def setup_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the pipeline.
    
    Args:
        config: Configuration dictionary containing directory paths
    """
    dirs = [
        config.get("checkpoint_dir", "data/checkpoints"),
        config.get("output_dir", "data/output"),
        config.get("log_dir", "logs")
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def save_checkpoint(file_path: str, data: Any, config: Dict[str, Any]) -> None:
    """
    Save checkpoint data to a file.
    
    Args:
        file_path: Path to save the checkpoint
        data: Data to be saved
        config: Configuration dictionary
    """
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    full_path = os.path.join(checkpoint_dir, file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Determine file extension and save accordingly
    if file_path.endswith('.json'):
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif file_path.endswith('.pkl'):
        import pickle
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported checkpoint file format: {file_path}")
    
    logger.info(f"Saved checkpoint: {full_path}")


def load_checkpoint(file_path: str, config: Dict[str, Any]) -> Any:
    """
    Load checkpoint data from a file.
    
    Args:
        file_path: Path to the checkpoint file
        config: Configuration dictionary
    
    Returns:
        Loaded data
    """
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    full_path = os.path.join(checkpoint_dir, file_path)
    
    if not os.path.exists(full_path):
        logger.warning(f"Checkpoint file not found: {full_path}")
        return None
    
    # Determine file extension and load accordingly
    if full_path.endswith('.json'):
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif full_path.endswith('.pkl'):
        import pickle
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported checkpoint file format: {full_path}")


def get_input_files(input_dir: str, file_pattern: str = "*.csv") -> List[str]:
    """
    Get all input CSV files from the specified directory.
    
    Args:
        input_dir: Directory containing input files
        file_pattern: File pattern to match (default: "*.csv")
    
    Returns:
        List of file paths
    """
    file_paths = sorted(str(p) for p in Path(input_dir).glob(file_pattern))
    logger.info(f"Found {len(file_paths)} input files in {input_dir}")
    return file_paths


def normalize_string(text: str) -> str:
    """
    Normalize a string by trimming whitespace and converting to lowercase.
    
    Args:
        text: Input string
    
    Returns:
        Normalized string
    """
    if not text or text.lower() == 'null':
        return ""
    
    # Basic normalization: strip whitespace and convert to lowercase
    return text.strip()


def hash_string(text: str) -> str:
    """
    Generate MD5 hash for a string.
    
    Args:
        text: Input string
    
    Returns:
        MD5 hash of the input string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def preprocess_catalog_data(
    input_files: List[str], 
    config: Dict[str, Any], 
    dev_mode: bool = False
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, Dict[str, str]], Dict[str, Dict[str, int]]]:
    """
    Process input CSV files to extract and deduplicate fields.
    
    Args:
        input_files: List of CSV file paths
        config: Configuration dictionary
        dev_mode: If True, process only a subset of data for development
    
    Returns:
        Tuple of (unique_strings, string_counts, record_field_hashes, field_hash_mapping)
    """
    # Initialize data structures
    unique_strings = {}  # Hash → String value
    string_counts = {}   # Hash → Frequency count
    record_field_hashes = {}  # Record ID → {Field → Hash}
    field_hash_mapping = {}   # Hash → {Field → Count}
    
    # Fields to process
    fields_to_process = [
        'record', 'person', 'roles', 'title', 'attribution', 
        'provision', 'subjects', 'genres'
    ]
    
    # Fields that can be null
    nullable_fields = ['attribution', 'provision', 'subjects', 'genres']
    
    # Filter out ground truth file explicitly
    filtered_input_files = []
    for f in input_files:
        if 'ground_truth.csv' in f:
            logger.info(f"Excluding ground truth file from preprocessing: {f}")
        else:
            filtered_input_files.append(f)
    
    input_files = filtered_input_files
    
    # Limit files in dev mode
    if dev_mode:
        max_files = config.get("dev_max_files", 5)
        input_files = input_files[:max_files]
        logger.info(f"Running in dev mode with {len(input_files)} files")
    
    # Process each file
    for file_path in tqdm(input_files, desc="Processing files"):
        try:
            logger.info(f"Processing file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Limit rows in dev mode
                rows = list(reader)
                if dev_mode:
                    max_rows = config.get("dev_max_rows", 1000)
                    rows = rows[:max_rows]
                
                # Process each row
                for row in rows:
                    record_id = row.get('id')
                    if not record_id:
                        logger.warning(f"Row missing ID in {file_path}, skipping")
                        continue
                    
                    record_field_hashes[record_id] = {}
                    
                    # Process composite 'record' field first if present
                    if 'record' in row and row['record']:
                        record_value = normalize_string(row['record'])
                        record_hash = hash_string(record_value)
                        
                        if record_hash not in unique_strings:
                            unique_strings[record_hash] = record_value
                            string_counts[record_hash] = 0
                        
                        string_counts[record_hash] += 1
                        record_field_hashes[record_id]['record'] = record_hash
                        
                        if record_hash not in field_hash_mapping:
                            field_hash_mapping[record_hash] = {}
                        
                        if 'record' not in field_hash_mapping[record_hash]:
                            field_hash_mapping[record_hash]['record'] = 0
                            
                        field_hash_mapping[record_hash]['record'] += 1
                    
                    # Process each field
                    for field in fields_to_process:
                        if field == 'record':  # Already processed
                            continue
                            
                        if field in row and row[field] and row[field].strip().lower() != 'null':
                            # Normalize and hash field value
                            field_value = normalize_string(row[field])
                            field_hash = hash_string(field_value)
                            
                            # Update data structures
                            if field_hash not in unique_strings:
                                unique_strings[field_hash] = field_value
                                string_counts[field_hash] = 0
                            
                            string_counts[field_hash] += 1
                            record_field_hashes[record_id][field] = field_hash
                            
                            # Update field/hash mapping
                            if field_hash not in field_hash_mapping:
                                field_hash_mapping[field_hash] = {}
                            
                            if field not in field_hash_mapping[field_hash]:
                                field_hash_mapping[field_hash][field] = 0
                                
                            field_hash_mapping[field_hash][field] += 1
                        elif field in nullable_fields:
                            # Mark nullable fields as NULL
                            record_field_hashes[record_id][field] = "NULL"
                        else:
                            # For required fields that are missing, log a warning
                            logger.warning(f"Required field '{field}' missing in record {record_id}")
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Save checkpoints
    if config.get("save_checkpoints", True):
        save_checkpoint('unique_strings.json', unique_strings, config)
        save_checkpoint('string_counts.json', string_counts, config)
        save_checkpoint('record_field_hashes.json', record_field_hashes, config)
        save_checkpoint('field_hash_mapping.json', field_hash_mapping, config)
    
    # Log stats
    logger.info(f"Processed data: {len(unique_strings)} unique strings, {len(record_field_hashes)} records")
    
    return unique_strings, string_counts, record_field_hashes, field_hash_mapping


def load_ground_truth(file_path: str) -> List[Tuple[str, str, bool]]:
    """
    Load ground truth data for training and evaluation.
    
    Args:
        file_path: Path to the ground truth CSV file
    
    Returns:
        List of (left_id, right_id, is_match) tuples
    """
    ground_truth = []
    
    if not os.path.exists(file_path):
        logger.error(f"Ground truth file not found: {file_path}")
        return ground_truth
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Clean up lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Skip the header
        if lines and len(lines) > 0:
            header = lines[0]
            logger.info(f"Ground truth header: {header}")
            
            # Process data rows
            for line in lines[1:]:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) != 3:
                    logger.warning(f"Skipping invalid ground truth row: {line}")
                    continue
                    
                left_id, right_id, match_str = parts
                is_match = match_str.lower() == 'true'
                ground_truth.append((left_id, right_id, is_match))
    except Exception as e:
        logger.error(f"Error loading ground truth from {file_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info(f"Loaded {len(ground_truth)} ground truth pairs")
    return ground_truth


if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Preprocessing module loaded successfully")
