#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification module for the entity resolution pipeline.
Handles feature engineering, classifier training, and entity clustering.
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Set
import time
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import Levenshtein
import networkx as nx
import numpy as np
import openai
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from utils import (
    cosine_similarity, normalize_features, extract_years, 
    save_checkpoint, save_output
)
from indexing import impute_null_field, get_candidates
from enhanced_features import (
    enhance_feature_vector,
    generate_high_value_interaction_features
)
# Try to import enhanced date processing integration
try:
    from integration import (
        check_enhanced_date_processing_available,
        update_config_for_enhanced_dates,
        extract_years_from_text,
        get_life_dates
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class LogisticRegressionClassifier:
    """
    Logistic regression classifier implemented with gradient descent.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        regularization: float = 0.01
    ):
        """
        Initialize the classifier.
        
        Args:
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations for training
            regularization: L2 regularization parameter
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = 0.0
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid function to input.
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid values
        """
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier using gradient descent.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        num_samples, num_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        
        # Store costs for analysis
        costs = []
        
        # Gradient descent
        for i in tqdm(range(self.num_iterations), desc="Training classifier"):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute cost
            cost = -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
            
            # Add L2 regularization
            cost += (self.regularization / (2 * num_samples)) * np.sum(np.square(self.weights))
            costs.append(cost)
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            dw += (self.regularization / num_samples) * self.weights  # L2 regularization gradient
            db = (1 / num_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Log progress
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Iteration {i+1}/{self.num_iterations}, Cost: {cost:.6f}")
        
        # Log final weights
        logger.info(f"Final weights: {self.weights}")
        logger.info(f"Final bias: {self.bias}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Predicted labels
        """
        return (self.predict_proba(X) >= threshold).astype(int)

def bulk_impute_fields(
    record_field_hashes: Dict[str, Dict[str, str]],
    weaviate_client: Any,
    config: Dict[str, Any]
) -> Dict[Tuple[str, str], List[float]]:
    """
    Pre-impute fields in bulk to avoid repeated queries.
    
    Args:
        record_field_hashes: Record field hashes
        weaviate_client: Weaviate client
        config: Configuration
        
    Returns:
        Dictionary mapping (record_hash, field) to imputed vector
    """

    # Skip bulk imputation if configured
    if config.get("skip_bulk_imputation", False):
        logger.info("Skipping bulk imputation (disabled in config)")
        return {}
    
    logger.info("Pre-imputing nullable fields...")
    
    from collections import defaultdict
    
    # Count field/hash combinations to find most common
    field_hash_counts = defaultdict(int)
    for fields in record_field_hashes.values():
        record_hash = fields.get('record', "NULL")
        if record_hash != "NULL":
            for field in ['attribution', 'provision', 'subjects', 'genres', 'relatedWork']:
                if fields.get(field) == "NULL":
                    field_hash_counts[(record_hash, field)] += 1
    
    # Get most common combinations
    common_combinations = sorted(
        field_hash_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:1000]  # Limit to 1000 most common
    
    # Pre-impute these combinations
    imputation_cache = {}
    logger.info(f"Pre-imputing {len(common_combinations)} common field/hash combinations")
    
    for (record_hash, field), _ in tqdm(common_combinations):
        vector = impute_null_field(record_hash, field, weaviate_client)
        if vector is not None:
            imputation_cache[(record_hash, field)] = vector
    
    logger.info(f"Completed pre-imputation of {len(imputation_cache)} combinations")
    return imputation_cache

def process_record_pair(
    pair_data: Tuple[str, str, bool],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    embeddings: Dict[str, List[float]],
    imputation_cache: Dict[Tuple[str, str], List[float]],
    weaviate_client: Any,
    config: Dict[str, Any],
    fields: List[str],
    nullable_fields: List[str],
    feature_names: List[str],
    use_enhanced_features: bool = False
) -> Optional[Tuple[List[float], int, List[str]]]:
    """
    Process a single record pair to generate features.
    
    Args:
        pair_data: Tuple of (id1, id2, label)
        record_field_hashes: Dictionary of record ID → {field → hash}
        unique_strings: Dictionary of hash → string value
        embeddings: Dictionary of hash → embedding vector
        imputation_cache: Cache for imputed values
        weaviate_client: Weaviate client
        config: Configuration dictionary
        fields: Fields to process
        nullable_fields: Fields that can be null
        feature_names: Names of features
        use_enhanced_features: Whether to use enhanced features
        
    Returns:
        Tuple of (feature_vector, label, enhanced_feature_names) or None if skipped
    """
    id1, id2, label = pair_data
    
    # Skip if either record is missing
    if id1 not in record_field_hashes or id2 not in record_field_hashes:
        return None
    
    # Get field hashes for both records
    fields1 = record_field_hashes.get(id1, {})
    fields2 = record_field_hashes.get(id2, {})
    
    # Initialize feature vector
    features = []
    
    # Process each field for similarity features
    field_vectors1 = []
    field_vectors2 = []
    field_indices = []
    
    local_imputation_cache = {}  # Thread-local cache
    
    # First collect all vectors
    for i, field in enumerate(fields):
        hash1 = fields1.get(field, "NULL")
        hash2 = fields2.get(field, "NULL")
        
        vector1 = None
        vector2 = None
        
        # Handle normal case
        if hash1 != "NULL":
            vector1 = embeddings.get(hash1)
        
        if hash2 != "NULL":
            vector2 = embeddings.get(hash2)
        
        # Handle imputation for nullable fields
        if hash1 == "NULL" and field in nullable_fields:
            # Check cache first
            record_hash = fields1.get('record', "NULL")
            cache_key = (record_hash, field)
            
            if cache_key in imputation_cache:
                vector1 = imputation_cache[cache_key]
            elif cache_key in local_imputation_cache:
                vector1 = local_imputation_cache[cache_key]
            else:
                vector1 = impute_null_field(record_hash, field, weaviate_client)
                if vector1 is not None:
                    local_imputation_cache[cache_key] = vector1
        
        if hash2 == "NULL" and field in nullable_fields:
            # Check cache first
            record_hash = fields2.get('record', "NULL")
            cache_key = (record_hash, field)
            
            if cache_key in imputation_cache:
                vector2 = imputation_cache[cache_key]
            elif cache_key in local_imputation_cache:
                vector2 = local_imputation_cache[cache_key]
            else:
                vector2 = impute_null_field(record_hash, field, weaviate_client)
                if vector2 is not None:
                    local_imputation_cache[cache_key] = vector2
        
        # If both vectors available, add to batch
        if vector1 is not None and vector2 is not None:
            field_vectors1.append(vector1)
            field_vectors2.append(vector2)
            field_indices.append(i)
        
        # Initialize with zeros
        features.append(0.0)
    
    # Batch compute similarities
    if field_vectors1:
        # Import here in case added to utils.py
        from utils import batch_cosine_similarity
        similarities = batch_cosine_similarity(field_vectors1, field_vectors2)
        
        # Update feature vector with computed similarities
        for i, sim_idx in enumerate(field_indices):
            features[sim_idx] = similarities[i]
    
    # Add special features for person names
    person_str1 = unique_strings.get(fields1.get('person', "NULL"), "")
    person_str2 = unique_strings.get(fields2.get('person', "NULL"), "")
    
    # Levenshtein distance (normalized)
    if person_str1 and person_str2:
        lev_distance = Levenshtein.distance(person_str1, person_str2)
        max_len = max(len(person_str1), len(person_str2))
        norm_lev = 1.0 - (lev_distance / max_len if max_len > 0 else 0.0)
        features.append(norm_lev)
    else:
        features.append(0.0)
    
    # Check for exact match with life dates (strong signal)
    has_life_dates1 = bool(re.search(r'\d{4}-\d{4}', person_str1))
    has_life_dates2 = bool(re.search(r'\d{4}-\d{4}', person_str2))
    exact_match_with_dates = person_str1 == person_str2 and (has_life_dates1 or has_life_dates2)
    features.append(1.0 if exact_match_with_dates else 0.0)
    
    # Add basic temporal overlap feature
    prov_str1 = unique_strings.get(fields1.get('provision', "NULL"), "")
    prov_str2 = unique_strings.get(fields2.get('provision', "NULL"), "")
    
    # Basic year extraction
    years1 = extract_years(prov_str1)
    years2 = extract_years(prov_str2)
    
    # Add temporal overlap indicator
    if years1 and years2:
        has_overlap = any(y1 in years2 for y1 in years1)
        features.append(1.0 if has_overlap else 0.0)
    else:
        features.append(0.5)  # Neutral when years can't be determined
        
    # Current feature names (without enhancement)
    current_feature_names = feature_names.copy()
    
    # Determine which type of feature enhancement to use
    use_enhanced_features = config.get("use_enhanced_features", False)
    interaction_features_only = config.get("interaction_features_only", False)
    
    if interaction_features_only:
        # Add only interaction features
        try:
            features, current_feature_names = add_interaction_features_only(
                features, 
                current_feature_names,
                unique_strings, 
                record_field_hashes, 
                id1, 
                id2,
                config
            )
        except Exception as e:
            logger.warning(f"Error adding interaction features: {e}")
    elif use_enhanced_features:
        # Use the full feature enhancement
        try:
            result = enhance_feature_vector(
                features, 
                current_feature_names,
                unique_strings, 
                record_field_hashes, 
                id1, 
                id2
            )
            
            # Check if result is valid
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                enhanced_features, enhanced_names = result
                features = enhanced_features
                current_feature_names = enhanced_names
            else:
                logger.warning(f"Invalid result from enhance_feature_vector: {result}")
        except Exception as e:
            # Log error but continue with basic features
            logger.warning(f"Error enhancing features: {e}")
    
    # Normalize feature vector if required
    if config.get("feature_normalization", True):
        features = normalize_features(features)
    
    return features, 1 if label else 0, current_feature_names

def engineer_features(
    record_pairs: List[Tuple[str, str, bool]],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    embeddings: Dict[str, List[float]],
    weaviate_client: Any,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate feature vectors for record pairs using parallel processing.

     Returns:
        Tuple of (X, y, feature_names)
    """

    # Start overall timing
    total_start_time = time.time()
    logger.info("====== STARTING FEATURE ENGINEERING ======")
    
    # Update config if integration module is available
    # if INTEGRATION_AVAILABLE:
    #     config = update_config_for_enhanced_dates(config)

    # Override configuration to disable enhanced date processing
    config = config.copy()  # Create a copy to avoid modifying the original
    config["use_enhanced_date_processing"] = False
    config["enhanced_temporal_features"] = False
    config["robust_date_handling"] = False
    
    # Keep interaction features enabled
    config["use_enhanced_features"] = False
    config["enhanced_feature_interactions"] = True
    
    # Initialize profiling dictionaries
    timings = defaultdict(float)
    counts = defaultdict(int)
    
    # Fields to process
    fields = ['person', 'record', 'title', 'roles', 'attribution', 
              'provision', 'subjects', 'genres', 'relatedWork']
    
    # Nullable fields
    nullable_fields = config.get("nullable_fields_to_use", 
                                ['attribution', 'provision', 'subjects', 'genres', 'relatedWork'])
    
    # Use enhanced features if enabled
    use_enhanced_features = config.get("use_enhanced_features", True)
    
    # Pre-impute common null fields
    logger.info("Performing bulk imputation for common fields...")
    imputation_start = time.time()
    imputation_cache = bulk_impute_fields(record_field_hashes, weaviate_client, config)
    timings['bulk_imputation'] = time.time() - imputation_start
    logger.info(f"Bulk imputation complete with {len(imputation_cache)} cached vectors in {timings['bulk_imputation']:.2f}s")
    
    # Create base feature names
    base_names = [f"{field}_sim" for field in fields]
    base_names.extend(['person_lev_sim', 'has_life_dates', 'temporal_overlap'])
    feature_names = base_names
    
    # Maximum number of threads
    max_workers = config.get("parallel_workers", min(8, os.cpu_count() or 4))
    logger.info(f"Processing {len(record_pairs)} record pairs using {max_workers} parallel workers")
    
    # Create a thread-local Weaviate client for each worker
    # This avoids connection closed errors when multiple threads use the same client
    def get_worker_client():
        return connect_to_weaviate(config)
    
    # Process in parallel
    processed_pairs = []
    skipped_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Submit all tasks
        for pair in record_pairs:
            future = executor.submit(
                process_record_pair,
                pair,
                record_field_hashes,
                unique_strings,
                embeddings,
                imputation_cache,
                weaviate_client,
                config,
                fields,
                nullable_fields,
                feature_names,
                use_enhanced_features
            )
            futures.append(future)
        
        # Process results with progress bar
        for future in tqdm(futures, desc="Engineering features"):
            result = future.result()
            if result is not None:
                features, label, enhanced_names = result
                processed_pairs.append((features, label))
                
                # Update feature names if this is the first processed pair with enhanced features
                if use_enhanced_features and len(feature_names) < len(enhanced_names):
                    feature_names = enhanced_names
            else:
                skipped_count += 1
    
    # Split results into X and y
    logger.info(f"Processed {len(processed_pairs)} pairs ({skipped_count} pairs skipped)")
    
    if not processed_pairs:
        logger.error("No valid pairs processed!")
        return np.array([]), np.array([])
    
    X = [pair[0] for pair in processed_pairs]
    y = [pair[1] for pair in processed_pairs]
    
    # Ensure all feature vectors have the same length (if using enhanced features)
    if use_enhanced_features and len(X) > 0:
        max_length = max(len(features) for features in X)
        
        # Pad shorter vectors
        for i in range(len(X)):
            if len(X[i]) < max_length:
                X[i] = X[i] + [0.0] * (max_length - len(X[i]))
                
        # Ensure feature names list matches
        if len(feature_names) < max_length:
            for i in range(len(feature_names), max_length):
                feature_names.append(f"feature_{i}")
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Log timing statistics
    logger.info("====== FEATURE ENGINEERING PROFILING RESULTS ======")
    logger.info(f"Total time: {total_time:.2f}s for {len(record_pairs)} pairs ({len(record_pairs)/total_time:.1f} pairs/sec)")
    logger.info(f"Bulk imputation time: {timings['bulk_imputation']:.2f}s")
    
    # Log feature dimensions
    logger.info(f"Generated {len(X)} feature vectors with {len(feature_names)} features each")
    
    # Store feature names in config for later use
    config["feature_names"] = feature_names
    
    logger.info("====== FEATURE ENGINEERING COMPLETE ======")
    
    # Convert to numpy arrays
    return np.array(X), np.array(y), feature_names

def train_classifier(
    X: np.ndarray, 
    y: np.ndarray, 
    config: Dict[str, Any]
) -> LogisticRegressionClassifier:
    """
    Train logistic regression classifier using gradient descent.
    
    Args:
        X: Feature matrix
        y: Labels
        config: Configuration dictionary
        
    Returns:
        Trained classifier
    """
    # Split data into train/test sets
    logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
    
    # Initialize and train classifier
    classifier = LogisticRegressionClassifier(
        learning_rate=config.get("learning_rate", 0.01),
        num_iterations=config.get("num_iterations", 1000),
        regularization=config.get("regularization", 0.01)
    )
    logger.info(f"Feature vector dimension: {X.shape[1]}")
    
    # Initialize and train classifier
    classifier = LogisticRegressionClassifier(
        learning_rate=config.get("learning_rate", 0.01),
        num_iterations=config.get("num_iterations", 1000),
        regularization=config.get("regularization", 0.01)
    )
    
    classifier.fit(X, y)
    
    # Evaluate on test set
    # y_pred = classifier.predict(X_test)
    
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    
    # logger.info(f"Test performance: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Save model
    save_checkpoint("classifier.pkl", classifier, config)
    
    # Save feature importance analysis
    feature_names = config.get("feature_names", None)
    if not feature_names:
        # Use default feature names if not provided
        fields = ['person', 'record', 'title', 'roles', 'attribution', 
                  'provision', 'subjects', 'genres', 'relatedWork']
        feature_names = [f"{field}_sim" for field in fields]
        feature_names.extend(['person_lev_sim', 'has_life_dates', 'temporal_overlap'])
        
        # Check if feature dimensions match
        if len(feature_names) != len(classifier.weights):
            # Create generic names if dimensions don't match
            feature_names = [f"feature_{i}" for i in range(len(classifier.weights))]
    
    feature_importance = sorted(
        zip(feature_names, classifier.weights),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    logger.info("Feature importance:")
    for feature, weight in feature_importance:
        logger.info(f"  {feature}: {weight:.4f}")
    
    return classifier


def llm_fallback(
    record1: str,
    record2: str,
    client: openai.OpenAI,
    config: Dict[str, Any]
) -> Optional[float]:
    """
    Use LLM to resolve ambiguous cases.
    
    Args:
        record1: First record text
        record2: Second record text
        client: OpenAI client
        config: Configuration dictionary
        
    Returns:
        Confidence score (1.0 for match, 0.0 for non-match, None for ambiguous)
    """
    # Check if LLM fallback is enabled
    if not config.get("use_llm_fallback", False):
        return None
    
    # Check request counter
    global_vars = globals()
    if "llm_request_counter" not in global_vars:
        global_vars["llm_request_counter"] = 0
        
    if global_vars["llm_request_counter"] >= config.get("max_llm_requests", 1000):
        logger.warning("Maximum LLM requests reached")
        return None
    
    # Increment counter
    global_vars["llm_request_counter"] += 1
    
    # Prepare prompts
    system_prompt = "You are a classifier deciding whether two records refer to the same person."
    user_prompt = f"""Tell me whether the following two records refer to the same person using a chain of reasoning followed by a single "yes" or "no" answer on one line, without any formatting.

1. {record1}

2. {record2}
"""
    
    try:
        # Query OpenAI
        response = client.chat.completions.create(
            model=config.get("llm_model", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        # Extract reasoning and decision
        lines = response_text.strip().split('\n')
        decision = lines[-1].strip().lower()
        
        logger.info(f"LLM decision for fallback: {decision}")
        
        if decision == "yes":
            return 1.0  # Same person
        elif decision == "no":
            return 0.0  # Different person
        else:
            logger.warning(f"Ambiguous LLM response: {decision}")
            return None  # Ambiguous
    except Exception as e:
        logger.error(f"Error in LLM fallback: {e}")
        return None


def derive_canonical_name(
    community: Set[str],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    config: Dict[str, Any] = None
) -> str:
    """
    Derive canonical name for a community of records.
    
    Args:
        community: Set of record IDs
        record_field_hashes: Dictionary of record ID → {field → hash}
        unique_strings: Dictionary of hash → string value
        config: Configuration dictionary (optional)
        
    Returns:
        Canonical name
    """
    # Default config
    if config is None:
        config = {}
    
    # Collect all person names in the community
    person_names = []
    person_with_dates = []
    
    for record_id in community:
        person_hash = record_field_hashes.get(record_id, {}).get('person', "NULL")
        if person_hash != "NULL":
            person_name = unique_strings.get(person_hash, "")
            person_names.append(person_name)
            
            # Check for life dates
            use_enhanced = config.get("use_enhanced_date_processing", False) and INTEGRATION_AVAILABLE
            
            if use_enhanced:
                birth_year, death_year, confidence = get_life_dates(person_name, True)
                if birth_year is not None or death_year is not None:
                    person_with_dates.append((person_name, confidence))
            else:
                if re.search(r'\d{4}-\d{4}', person_name):
                    person_with_dates.append((person_name, 1.0))
    
    if not person_names:
        return "Unknown Person"
    
    # Prefer names with life dates
    if person_with_dates:
        # Use the name with highest confidence if enhanced dates are used
        if config.get("use_enhanced_date_processing", False) and INTEGRATION_AVAILABLE:
            # Sort by confidence (descending)
            person_with_dates.sort(key=lambda x: x[1], reverse=True)
            return person_with_dates[0][0]
        else:
            # Return the most detailed name with dates
            return max((name for name, _ in person_with_dates), key=len)
    else:
        # Return the most frequent name
        name_counts = {}
        for name in person_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        most_common = max(name_counts.items(), key=lambda x: x[1])
        return most_common[0]


def calculate_cluster_confidence(
    community: Set[str],
    graph: nx.Graph
) -> float:
    """
    Calculate confidence score for a cluster.
    
    Args:
        community: Set of record IDs
        graph: NetworkX graph with edge weights
        
    Returns:
        Confidence score (0.0 - 1.0)
    """
    if len(community) <= 1:
        return 1.0  # Single node communities are perfectly confident
    
    # Get all edges within the community
    community_edges = []
    for u in community:
        for v in community:
            if u != v and graph.has_edge(u, v):
                weight = graph.get_edge_data(u, v).get('weight', 0.0)
                community_edges.append(weight)
    
    if not community_edges:
        return 0.0
    
    # Return the average edge weight
    return sum(community_edges) / len(community_edges)


def classify_and_cluster(
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    embeddings: Dict[str, List[float]],
    classifier: LogisticRegressionClassifier,
    weaviate_client: Any,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Classify record pairs and cluster entities using ANN-based blocking with person vectors.
    
    This implementation uses Approximate Nearest Neighbor (ANN) search to find similar
    person vectors, effectively using the 'person' field as a blocking key to limit
    the number of pairwise comparisons.
    """
    # Update config for enhanced date processing if available
    if INTEGRATION_AVAILABLE:
        config = update_config_for_enhanced_dates(config)

    # Get all record IDs
    record_ids = list(record_field_hashes.keys())
    logger.info(f"Processing {len(record_ids)} records for classification")
    
    # Initialize graph for clustering
    G = nx.Graph()
    
    # Add all nodes to the graph
    for record_id in record_ids:
        G.add_node(record_id)
    
    # Set up OpenAI client for LLM fallback if enabled
    llm_client = None
    if config.get("use_llm_fallback", False):
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            llm_client = openai.OpenAI(api_key=api_key)
            logger.info("LLM fallback enabled")
        else:
            logger.warning("LLM fallback enabled but OpenAI API key not found")
    
    # Process records in batches
    batch_size = config.get("classification_batch_size", 1000)
    confidence_threshold = config.get("confidence_threshold", 0.7)
    candidate_limit = config.get("candidate_limit", 100)
    
    # Create a lookup from person hash to record IDs for efficient candidate matching
    logger.info("Building person hash to record mapping")
    person_hash_to_records = defaultdict(list)
    for record_id, fields in tqdm(record_field_hashes.items(), desc="Building index"):
        person_hash = fields.get('person', "NULL")
        if person_hash != "NULL":
            person_hash_to_records[person_hash].append(record_id)
    
    processed_pairs = set()  # Track processed pairs to avoid duplicates
    
    for i in range(0, len(record_ids), batch_size):
        batch_ids = record_ids[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(record_ids)-1)//batch_size + 1}")
        
        # Collect candidate pairs for this batch
        batch_candidate_pairs = []
        
        # Process each record in the batch to find candidates using ANN
        for record_id in tqdm(batch_ids, desc="Finding candidates via ANN"):
            # Get person hash for this record
            person_hash = record_field_hashes.get(record_id, {}).get('person', "NULL")
            if person_hash == "NULL":
                continue
            
            # Get person vector
            person_vector = embeddings.get(person_hash)
            if person_vector is None:
                continue
            
            # Query Weaviate for similar person vectors (ANN-based blocking)
            candidates = get_candidates(
                person_vector=person_vector,
                client=weaviate_client,
                limit=candidate_limit
            )
            
            # Get candidate record IDs directly from the candidate hashes
            for candidate in candidates:
                candidate_hash = candidate['hash']
                # Skip if it's the same hash as the current record
                if candidate_hash == person_hash:
                    continue
                
                # Get records with this person hash using our lookup map
                candidate_records = person_hash_to_records.get(candidate_hash, [])
                for candidate_record_id in candidate_records:
                    # Skip self comparisons
                    if candidate_record_id == record_id:
                        continue
                    
                    # Create a candidate pair
                    pair_key = tuple(sorted([record_id, candidate_record_id]))
                    
                    # Skip if this pair has been processed
                    if pair_key in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_key)
                    batch_candidate_pairs.append((record_id, candidate_record_id, None))
        
        # Skip if no pairs to process
        if not batch_candidate_pairs:
            logger.info("No candidate pairs to process in this batch")
            continue
            
        # Generate features and classify all pairs at once
        logger.info(f"Classifying {len(batch_candidate_pairs)} candidate pairs")
        X, _, _ = engineer_features(
            batch_candidate_pairs,
            record_field_hashes,
            unique_strings,
            embeddings,
            weaviate_client,
            config
        )
        
        # Classify all pairs
        probabilities = classifier.predict_proba(X)
        
        # Process classification results
        for idx, ((id1, id2, _), probability) in enumerate(zip(batch_candidate_pairs, probabilities)):
            # Add edge to graph if probability exceeds threshold
            if probability >= confidence_threshold:
                G.add_edge(id1, id2, weight=probability)
            else:
                # Check for exact match with life dates (strong signal)
                person_str1 = unique_strings.get(record_field_hashes.get(id1, {}).get('person', "NULL"), "")
                person_str2 = unique_strings.get(record_field_hashes.get(id2, {}).get('person', "NULL"), "")
                
                has_life_dates1 = bool(re.search(r'\d{4}-\d{4}', person_str1))
                has_life_dates2 = bool(re.search(r'\d{4}-\d{4}', person_str2))
                exact_match_with_dates = person_str1 == person_str2 and (has_life_dates1 or has_life_dates2)
                
                if exact_match_with_dates:
                    # Override classifier decision
                    G.add_edge(id1, id2, weight=0.95)
                elif (0.5 <= probability < confidence_threshold) and llm_client:
                    # Ambiguous case, try LLM fallback
                    record1 = unique_strings.get(record_field_hashes.get(id1, {}).get('record', "NULL"), "")
                    record2 = unique_strings.get(record_field_hashes.get(id2, {}).get('record', "NULL"), "")
                    
                    llm_result = llm_fallback(record1, record2, llm_client, config)
                    
                    if llm_result is not None and llm_result > 0.5:
                        G.add_edge(id1, id2, weight=llm_result)
    
    # Apply community detection for clustering
    logger.info("Applying community detection algorithm")
    try:
        communities = nx.community.louvain_communities(G, weight='weight')
        logger.info(f"Found {len(communities)} communities")
    except Exception as e:
        logger.error(f"Error in community detection: {e}")
        logger.info("Falling back to connected components")
        communities = list(nx.connected_components(G))
        logger.info(f"Found {len(communities)} connected components")
    
    # Format clusters
    entity_clusters = []
    for i, community in enumerate(communities):
        # Skip large communities (likely false positives)
        max_community_size = config.get("max_community_size", 10000)
        if len(community) > max_community_size:
            logger.warning(f"Skipping large community with {len(community)} members")
            continue
        
        canonical_name = derive_canonical_name(community, record_field_hashes, unique_strings, config)
        confidence = calculate_cluster_confidence(community, G)
        
        cluster = {
            "cluster_id": i,
            "canonical_name": canonical_name,
            "members": sorted(list(community)),
            "confidence": confidence,
            "size": len(community)
        }
        entity_clusters.append(cluster)
    
    # Sort clusters by size (descending)
    entity_clusters.sort(key=lambda x: x["size"], reverse=True)
    
    # Save clusters
    save_output("entity_clusters.jsonl", entity_clusters, config)
    
    logger.info(f"Created {len(entity_clusters)} entity clusters")
    return entity_clusters


def evaluate_clusters(
    entity_clusters: List[Dict[str, Any]],
    ground_truth_pairs: List[Tuple[str, str, bool]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate clustering results against ground truth.
    
    Args:
        entity_clusters: List of entity clusters
        ground_truth_pairs: List of (id1, id2, is_match) tuples
        config: Configuration dictionary
        
    Returns:
        Evaluation metrics
    """
    # Create a map from record ID to cluster ID
    record_to_cluster = {}
    for cluster in entity_clusters:
        cluster_id = cluster["cluster_id"]
        for record_id in cluster["members"]:
            record_to_cluster[record_id] = cluster_id
    
    # Evaluate each ground truth pair
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for id1, id2, is_match in ground_truth_pairs:
        # Skip if either record is not in any cluster
        if id1 not in record_to_cluster or id2 not in record_to_cluster:
            continue
        
        # Check if records are in the same cluster
        same_cluster = record_to_cluster[id1] == record_to_cluster[id2]
        
        if is_match and same_cluster:
            true_positives += 1
        elif is_match and not same_cluster:
            false_negatives += 1
        elif not is_match and same_cluster:
            false_positives += 1
        else:  # not is_match and not same_cluster
            true_negatives += 1
    
    # Calculate metrics
    total = true_positives + false_positives + true_negatives + false_negatives
    
    try:
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    except ZeroDivisionError:
        logger.error("Error calculating metrics: division by zero")
        precision, recall, f1, accuracy = 0, 0, 0, 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total_evaluated": total
    }
    
    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    
    # Save metrics
    save_output("evaluation_metrics.json", metrics, config)
    
    return metrics

def generate_detailed_reports(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_pairs: List[Tuple[str, str, bool]],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    feature_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate detailed reports on classifier performance.
    
    Args:
        X_test: Test feature matrix
        y_test: True labels
        y_pred: Predicted labels
        test_pairs: Original test record pairs
        record_field_hashes: Record field hashes
        unique_strings: Dictionary of hash → string value
        feature_names: Names of features
        config: Configuration dictionary
    """
    output_dir = config.get("output_dir", "data/output")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate report of misclassified pairs
    generate_misclassified_report(
        X_test, y_test, y_pred, test_pairs, 
        record_field_hashes, unique_strings, 
        feature_names, reports_dir
    )
    
    # Generate complete test dataset report
    generate_test_dataset_report(
        X_test, y_test, y_pred, test_pairs,
        record_field_hashes, unique_strings,
        feature_names, reports_dir
    )
    
    logger.info(f"Detailed reports saved to {reports_dir}")

def generate_misclassified_report(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_pairs: List[Tuple[str, str, bool]],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    feature_names: List[str],
    output_dir: str
) -> None:
    """
    Generate a CSV report of misclassified pairs.
    
    Args:
        X_test: Test feature matrix
        y_test: True labels
        y_pred: Predicted labels
        test_pairs: Original test record pairs
        record_field_hashes: Record field hashes
        unique_strings: Dictionary of hash → string value
        feature_names: Names of features
        output_dir: Directory to save report
    """
    misclassified_indices = np.where(y_test != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        logger.info("No misclassified pairs to report")
        return
    
    # Create CSV file
    csv_path = os.path.join(output_dir, "misclassified_pairs.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # Define headers
        writer = csv.writer(f)
        headers = [
            'record_id1', 'record_id2', 
            'person1', 'person2',
            'title1', 'title2',
            'provision1', 'provision2',
            'true_label', 'predicted_label'
        ]
        # Add feature names to headers
        headers.extend(feature_names)
        
        writer.writerow(headers)
        
        # Write rows for each misclassified pair
        for idx in misclassified_indices:
            if idx < len(test_pairs):
                id1, id2, _ = test_pairs[idx]
                
                # Get field values
                fields1 = record_field_hashes.get(id1, {})
                fields2 = record_field_hashes.get(id2, {})
                
                person1 = unique_strings.get(fields1.get('person', "NULL"), "")
                person2 = unique_strings.get(fields2.get('person', "NULL"), "")
                
                title1 = unique_strings.get(fields1.get('title', "NULL"), "")
                title2 = unique_strings.get(fields2.get('title', "NULL"), "")
                
                provision1 = unique_strings.get(fields1.get('provision', "NULL"), "")
                provision2 = unique_strings.get(fields2.get('provision', "NULL"), "")
                
                # Create row
                row = [
                    id1, id2,
                    person1, person2,
                    title1, title2,
                    provision1, provision2,
                    int(y_test[idx]), int(y_pred[idx])
                ]
                
                # Add feature values
                row.extend(X_test[idx].tolist())
                
                writer.writerow(row)
    
    logger.info(f"Saved report of {len(misclassified_indices)} misclassified pairs to {csv_path}")

def generate_test_dataset_report(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_pairs: List[Tuple[str, str, bool]],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    feature_names: List[str],
    output_dir: str
) -> None:
    """
    Generate a complete report of the test dataset with feature vectors.
    
    Args:
        X_test: Test feature matrix
        y_test: True labels
        y_pred: Predicted labels
        test_pairs: Original test record pairs
        record_field_hashes: Record field hashes
        unique_strings: Dictionary of hash → string value
        feature_names: Names of features
        output_dir: Directory to save report
    """
    # Create CSV file
    csv_path = os.path.join(output_dir, "test_dataset_complete.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # Define headers
        writer = csv.writer(f)
        headers = [
            'record_id1', 'record_id2', 
            'person1', 'person2',
            'true_label', 'predicted_label', 'is_correct'
        ]
        # Add feature names to headers
        headers.extend(feature_names)
        
        writer.writerow(headers)
        
        # Write rows for each test pair
        for idx in range(len(test_pairs)):
            if idx < len(test_pairs):
                id1, id2, _ = test_pairs[idx]
                
                # Get field values
                fields1 = record_field_hashes.get(id1, {})
                fields2 = record_field_hashes.get(id2, {})
                
                person1 = unique_strings.get(fields1.get('person', "NULL"), "")
                person2 = unique_strings.get(fields2.get('person', "NULL"), "")
                
                # Create row
                row = [
                    id1, id2,
                    person1, person2,
                    int(y_test[idx]), int(y_pred[idx]),
                    int(y_test[idx] == y_pred[idx])
                ]
                
                # Add feature values
                if idx < len(X_test):
                    row.extend(X_test[idx].tolist())
                
                writer.writerow(row)
    
    logger.info(f"Saved complete test dataset report with {len(test_pairs)} pairs to {csv_path}")

def add_interaction_features_only(
    base_features: List[float],
    feature_names: List[str],
    unique_strings: Dict[str, str],
    record_field_hashes: Dict[str, Dict[str, str]],
    record_id1: str,
    record_id2: str,
    config: Dict[str, Any]
) -> Tuple[List[float], List[str]]:
    """
    Add only interaction features to the base feature vector without other enhancements.
    This function is used when you want interaction features but not other enhanced features.
    
    Args:
        base_features: Base feature vector
        feature_names: Names of base features
        unique_strings: Dictionary of hash → string value
        record_field_hashes: Dictionary of record ID → field → hash
        record_id1: First record ID
        record_id2: Second record ID
        config: Configuration dictionary
        
    Returns:
        Tuple of (enhanced_features, enhanced_feature_names)
    """
    # Convert base features to dictionary
    base_features_dict = {name: value for name, value in zip(feature_names, base_features)}
    
    # Generate only the interaction features
    interaction_features = generate_high_value_interaction_features(base_features_dict)
    
    # Combine with base features
    enhanced_features_dict = {**base_features_dict, **interaction_features}
    
    # Convert back to list format
    enhanced_feature_names = list(enhanced_features_dict.keys())
    enhanced_feature_values = [enhanced_features_dict[name] for name in enhanced_feature_names]
    
    return enhanced_feature_values, enhanced_feature_names

if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Classification module loaded successfully")
