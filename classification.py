#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification module for the entity resolution pipeline.
Handles feature engineering, classifier training, and entity clustering.
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Set

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
    generate_interaction_features
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


def engineer_features(
    record_pairs: List[Tuple[str, str, bool]],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    embeddings: Dict[str, List[float]],
    weaviate_client: Any,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate feature vectors for record pairs.
    
    Args:
        record_pairs: List of (id1, id2, label) tuples
        record_field_hashes: Dictionary of record ID → {field → hash}
        unique_strings: Dictionary of hash → string value
        embeddings: Dictionary of hash → embedding vector
        weaviate_client: Weaviate client
        config: Configuration dictionary
        
    Returns:
        Tuple of (X, y) feature matrix and labels
    """
    # Update config if integration module is available
    if INTEGRATION_AVAILABLE:
        config = update_config_for_enhanced_dates(config)
    
    X, y = [], []
    feature_names = []
    
    # Fields to process
    fields = ['person', 'record', 'title', 'roles', 'attribution', 
              'provision', 'subjects', 'genres', 'relatedWork']
    
    # Nullable fields
    nullable_fields = ['attribution', 'provision', 'subjects', 'genres', 'relatedWork']
    
    # Use enhanced features if enabled
    use_enhanced_features = config.get("use_enhanced_features", True)
    
    for id1, id2, label in tqdm(record_pairs, desc="Engineering features"):
        # Skip if either record is missing
        if id1 not in record_field_hashes or id2 not in record_field_hashes:
            logger.warning(f"Skipping pair with missing record: {id1}, {id2}")
            continue
        
        # Get field hashes for both records
        fields1 = record_field_hashes.get(id1, {})
        fields2 = record_field_hashes.get(id2, {})
        
        # Initialize feature vector
        features = []
        
        # Process each field for similarity features
        for field in fields:
            hash1 = fields1.get(field, "NULL")
            hash2 = fields2.get(field, "NULL")
            
            # Get vectors for both fields
            vector1 = None
            vector2 = None
            
            # Handle normal case
            if hash1 != "NULL":
                vector1 = embeddings.get(hash1)
            
            if hash2 != "NULL":
                vector2 = embeddings.get(hash2)
            
            # Handle imputation for nullable fields
            if hash1 == "NULL" and field in nullable_fields:
                record_hash = fields1.get('record', "NULL")
                vector1 = impute_null_field(record_hash, field, weaviate_client)
                
            if hash2 == "NULL" and field in nullable_fields:
                record_hash = fields2.get('record', "NULL")
                vector2 = impute_null_field(record_hash, field, weaviate_client)
            
            # Compute cosine similarity if both vectors are available
            if vector1 is not None and vector2 is not None:
                similarity = cosine_similarity(vector1, vector2)
            else:
                similarity = 0.0
                
            features.append(similarity)
        
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
        use_enhanced_dates = config.get("use_enhanced_date_processing", False) and INTEGRATION_AVAILABLE
        
        if use_enhanced_dates:
            # Use enhanced date extraction
            birth_year1, death_year1, conf1 = get_life_dates(person_str1, True)
            birth_year2, death_year2, conf2 = get_life_dates(person_str2, True)
            
            # Enhanced life dates match
            has_life_dates1 = birth_year1 is not None or death_year1 is not None
            has_life_dates2 = birth_year2 is not None or death_year2 is not None
            
            # Calculate enhanced match score with confidence
            if (birth_year1 == birth_year2 and birth_year1 is not None and 
                death_year1 == death_year2 and death_year1 is not None):
                life_dates_match = 1.0 * ((conf1 + conf2) / 2)  # Weight by confidence
            else:
                life_dates_match = 0.0
            
            features.append(1.0 if has_life_dates1 or has_life_dates2 else 0.0)
            features.append(life_dates_match)
        else:
            # Basic extraction (backward compatible)
            has_life_dates1 = bool(re.search(r'\d{4}-\d{4}', person_str1))
            has_life_dates2 = bool(re.search(r'\d{4}-\d{4}', person_str2))
            exact_match_with_dates = person_str1 == person_str2 and (has_life_dates1 or has_life_dates2)
            features.append(1.0 if exact_match_with_dates else 0.0)
        
        # Add basic temporal overlap feature (will be enhanced if enabled)
        prov_str1 = unique_strings.get(fields1.get('provision', "NULL"), "")
        prov_str2 = unique_strings.get(fields2.get('provision', "NULL"), "")
        
        if use_enhanced_dates:
            # Use enhanced year extraction
            years1 = extract_years_from_text(prov_str1, True)
            years2 = extract_years_from_text(prov_str2, True)
        else:
            # Basic year extraction
            years1 = extract_years(prov_str1)
            years2 = extract_years(prov_str2)
        
        # Add temporal overlap indicator
        if years1 and years2:
            has_overlap = any(y1 in years2 for y1 in years1)
            features.append(1.0 if has_overlap else 0.0)
        else:
            features.append(0.5)  # Neutral when years can't be determined
        
        # Create feature names if this is the first pair
        if not feature_names:
            base_names = [f"{field}_sim" for field in fields]
            
            if use_enhanced_dates:
                base_names.extend([
                    'person_lev_sim', 
                    'has_life_dates',
                    'life_dates_match_score',
                    'temporal_overlap'
                ])
            else:
                base_names.extend(['person_lev_sim', 'has_life_dates', 'temporal_overlap'])
                
            feature_names = base_names
        
        # Enhance features if enabled
        if use_enhanced_features:
            enhanced_features, enhanced_names = enhance_feature_vector(
                features, feature_names, unique_strings, record_field_hashes, id1, id2
            )
            # Update feature vector and names
            features = enhanced_features
            
            # Update feature names only on first pair
            if len(X) == 0:
                feature_names = enhanced_names
        
        # Normalize feature vector
        if config.get("feature_normalization", True):
            features = normalize_features(features)
        
        X.append(features)
        y.append(1 if label else 0)
    
    # Log feature names
    logger.info(f"Engineered features: {feature_names}")
    
    # Store feature names in config for later use
    config["feature_names"] = feature_names
    
    return np.array(X), np.array(y)

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
    random_seed = config.get("random_seed", 42)
    test_size = 1 - config.get("train_test_split", 0.8)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    logger.info(f"Feature vector dimension: {X_train.shape[1]}")
    
    # Initialize and train classifier
    classifier = LogisticRegressionClassifier(
        learning_rate=config.get("learning_rate", 0.01),
        num_iterations=config.get("num_iterations", 1000),
        regularization=config.get("regularization", 0.01)
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Test performance: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
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
    Classify record pairs and cluster entities.
    
    Args:
        record_field_hashes: Dictionary of record ID → {field → hash}
        unique_strings: Dictionary of hash → string value
        embeddings: Dictionary of hash → embedding vector
        classifier: Trained classifier
        weaviate_client: Weaviate client
        config: Configuration dictionary
        
    Returns:
        List of entity clusters
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
    
    processed_pairs = set()  # Track processed pairs to avoid duplicates
    
    for i in range(0, len(record_ids), batch_size):
        batch_ids = record_ids[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(record_ids)-1)//batch_size + 1}")
        
        # Process each record in the batch
        for record_id in tqdm(batch_ids, desc="Finding candidates"):
            # Get person hash for this record
            person_hash = record_field_hashes.get(record_id, {}).get('person', "NULL")
            if person_hash == "NULL":
                continue
            
            # Get person vector
            person_vector = embeddings.get(person_hash)
            if person_vector is None:
                continue
            
            # Query Weaviate for similar person vectors
            candidates = get_candidates(
                person_vector=person_vector,
                client=weaviate_client,
                limit=candidate_limit
            )
            
            # Get candidate record IDs
            candidate_hashes = [candidate['hash'] for candidate in candidates]
            
            # Find records with these person hashes
            candidate_ids = []
            for candidate_hash in candidate_hashes:
                for rid, fields in record_field_hashes.items():
                    if fields.get('person') == candidate_hash and rid != record_id:
                        candidate_ids.append(rid)
            
            # Classify each unique pair
            for candidate_id in candidate_ids:
                # Skip if this pair has been processed
                pair_key = tuple(sorted([record_id, candidate_id]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Generate feature vector for this pair
                features = engineer_features(
                    [(record_id, candidate_id, None)],  # Label doesn't matter here
                    record_field_hashes,
                    unique_strings,
                    embeddings,
                    weaviate_client,
                    config
                )[0]
                
                if len(features) == 0:
                    continue
                
                # Classify
                probability = float(classifier.predict_proba(features)[0])
                
                # Add edge to graph if probability exceeds threshold
                if probability >= confidence_threshold:
                    G.add_edge(record_id, candidate_id, weight=probability)
                else:
                    # Check for exact match with life dates (strong signal)
                    person_str1 = unique_strings.get(record_field_hashes.get(record_id, {}).get('person', "NULL"), "")
                    person_str2 = unique_strings.get(record_field_hashes.get(candidate_id, {}).get('person', "NULL"), "")
                    
                    has_life_dates1 = bool(re.search(r'\d{4}-\d{4}', person_str1))
                    has_life_dates2 = bool(re.search(r'\d{4}-\d{4}', person_str2))
                    exact_match_with_dates = person_str1 == person_str2 and (has_life_dates1 or has_life_dates2)
                    
                    if exact_match_with_dates:
                        # Override classifier decision
                        G.add_edge(record_id, candidate_id, weight=0.95)
                    elif (0.5 <= probability < confidence_threshold) and llm_client:
                        # Ambiguous case, try LLM fallback
                        record1 = unique_strings.get(record_field_hashes.get(record_id, {}).get('record', "NULL"), "")
                        record2 = unique_strings.get(record_field_hashes.get(candidate_id, {}).get('record', "NULL"), "")
                        
                        llm_result = llm_fallback(record1, record2, llm_client, config)
                        
                        if llm_result is not None and llm_result > 0.5:
                            G.add_edge(record_id, candidate_id, weight=llm_result)
    
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
        
        canonical_name = derive_canonical_name(community, record_field_hashes, unique_strings)
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


if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Classification module loaded successfully")
