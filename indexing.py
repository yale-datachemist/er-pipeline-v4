#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indexing module for the entity resolution pipeline.
Handles Weaviate integration for vector indexing and similarity search.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import weaviate
from tqdm import tqdm
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter

# Configure logging
logger = logging.getLogger(__name__)


def connect_to_weaviate(config: Dict[str, Any]) -> weaviate.Client:
    """
    Connect to Weaviate instance.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Weaviate client
    """
    url = config.get("weaviate_url", "http://localhost:8080")
    timeout = config.get("weaviate_timeout", 300)
    
    try:
        # client = weaviate.connect_to_custom(
        #     url=url,
        #     timeout_config=(timeout, timeout)  # (connect_timeout, read_timeout)
        # )
        client = weaviate.connect_to_local()
        logger.info(f"Connected to Weaviate at {url}")
        return client
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        raise


def create_schema(client: weaviate.Client, config: Dict[str, Any]) -> None:
    """
    Create Weaviate schema for entity resolution.
    
    Args:
        client: Weaviate client
        config: Configuration dictionary
    """
    # Check if collection exists
    if client.collections.exists("UniqueStringsByField"):
        logger.info("Collection 'UniqueStringsByField' already exists")
        return
    
    # Define named vectors for each field
    field_types = [
        "record", "person", "roles", "title", "attribution",
        "provision", "subjects", "genres", "relatedWork"
    ]
    
    vector_configs = []
    for field in field_types:
        vector_configs.append(
            Configure.NamedVectors.none(
                name=field,
                vector_index_config=Configure.VectorIndex.hnsw(
                    ef=config.get("weaviate_ef", 128),
                    maxConnections=config.get("weaviate_max_connections", 64),
                    efConstruction=config.get("weaviate_ef_construction", 128),
                    distance=config.get("weaviate_distance", "cosine")
                )
            )
        )
    
    # Create collection with schema
    client.collections.create(
        "UniqueStringsByField",
        vectorizer_config=vector_configs,
        properties=[
            Property(name="string_value", data_type=DataType.TEXT),
            Property(name="hash", data_type=DataType.TEXT, index_filterable=True),
            Property(name="frequency", data_type=DataType.NUMBER),
            Property(name="field_type", data_type=DataType.TEXT, index_filterable=True),
        ],
    )
    
    logger.info("Created 'UniqueStringsByField' collection with named vectors")


def index_embeddings(
    unique_strings: Dict[str, str],
    embeddings: Dict[str, List[float]],
    field_hash_mapping: Dict[str, Dict[str, int]],
    string_counts: Dict[str, int],
    config: Dict[str, Any]
) -> weaviate.Client:
    """
    Index embeddings in Weaviate for efficient similarity search.
    
    Args:
        unique_strings: Dictionary of hash → string value
        embeddings: Dictionary of hash → embedding vector
        field_hash_mapping: Dictionary of hash → {field → count}
        string_counts: Dictionary of hash → frequency count
        config: Configuration dictionary
        
    Returns:
        Weaviate client
    """
    # Connect to Weaviate
    client = connect_to_weaviate(config)
    
    # Create schema if needed
    create_schema(client, config)
    
    # Get collection
    collection = client.collections.get("UniqueStringsByField")
    
    # Check if data is already indexed
    if config.get("skip_if_indexed", False):
        try:
            # Check count of objects in collection
            count = collection.query.fetch_objects(limit=1).total_count
            if count > 0:
                logger.info(f"Found {count} objects already indexed, skipping indexing")
                return client
        except Exception as e:
            logger.warning(f"Error checking existing objects: {e}")
    
    # Count items to be indexed
    total_items = sum(
        len(fields) for h, fields in field_hash_mapping.items() 
        if h in embeddings
    )
    logger.info(f"Indexing {total_items} items in Weaviate")
    
    # Get batch size
    batch_size = config.get("weaviate_batch_size", 100)
    
    # Use batch processing for efficiency
    with collection.batch.dynamic() as batch:
        counter = 0
        for hash_value in tqdm(field_hash_mapping.keys(), desc="Indexing embeddings"):
            # Skip if this hash doesn't have an embedding
            if hash_value not in embeddings:
                continue
            
            # Get the string value and embedding
            string_value = unique_strings.get(hash_value, "")
            embedding_vector = embeddings.get(hash_value, [])
            
            # For each field type this string appears in
            for field_type, count in field_hash_mapping.get(hash_value, {}).items():
                # Create vector dictionary for named vectors
                vectors = {field_type: embedding_vector}
                
                # Insert object
                batch.add_object(
                    properties={
                        "string_value": string_value,
                        "hash": hash_value,
                        "frequency": string_counts.get(hash_value, 0),
                        "field_type": field_type,
                    },
                    vector=vectors
                )
                
                counter += 1
                
                # Log progress periodically
                if counter % (batch_size * 10) == 0:
                    logger.info(f"Indexed {counter} items")
    
    # Ensure indexing is complete
    logger.info("Waiting for indexing to complete...")
    collection.wait_for_indexing()
    
    logger.info(f"Successfully indexed {counter} items")
    return client


def vector_search(
    client: weaviate.Client,
    query_vector: List[float],
    field_type: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform vector search in Weaviate.
    
    Args:
        client: Weaviate client
        query_vector: Query vector
        field_type: Field type to search
        limit: Maximum number of results
    
    Returns:
        List of search results
    """
    collection = client.collections.get("UniqueStringsByField")
    
    try:
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            include_vector=True,
            filters=Filter.by_property("field_type").equal(field_type)
        )
        
        return [
            {
                "string_value": obj.properties.get("string_value", ""),
                "hash": obj.properties.get("hash", ""),
                "distance": obj.metadata.distance,
                "vector": obj.vector.get(field_type, []) if obj.vector else []
            }
            for obj in results.objects
        ]
    except Exception as e:
        logger.error(f"Error performing vector search: {e}")
        return []


def impute_null_field(
    record_hash: str,
    field_to_impute: str,
    client: weaviate.Client,
    k: int = 10
) -> Optional[List[float]]:
    """
    Impute missing field values using vector-based hot deck approach.
    
    Args:
        record_hash: Hash of the record field
        field_to_impute: Field to impute
        client: Weaviate client
        k: Number of nearest neighbors to consider
        
    Returns:
        Imputed vector or None if imputation fails
    """
    # Skip if record hash is NULL
    if record_hash == "NULL":
        logger.debug(f"Skipping imputation for NULL record hash")
        return None
    
    # Get the collection
    collection = client.collections.get("UniqueStringsByField")
    
    try:
        # First, get the record vector
        hash_filter = Filter.by_property("hash").equal(record_hash)
        field_filter = Filter.by_property("field_type").equal("record")
        
        # Combined filter
        combined_filter = hash_filter & field_filter
        
        result = collection.query.fetch_objects(
            filters=combined_filter,
            limit=1,
            include_vector=True
        )
        
        if not result.objects:
            logger.warning(f"No record found for hash {record_hash}")
            return None
        
        # Get the record vector
        record_vector = result.objects[0].vector.get('record')
        if not record_vector:
            logger.warning(f"No vector found for record hash {record_hash}")
            return None
        
        # Query for nearest vectors of the specific field type
        results = collection.query.near_vector(
            near_vector=record_vector,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
            include_vector=True,
            filters=Filter.by_property("field_type").equal(field_to_impute)
        )
        
        # Extract vectors from results
        vectors = []
        for obj in results.objects:
            vector = obj.vector.get(field_to_impute)
            if vector:
                vectors.append(vector)
        
        # Compute weighted average vector based on distance
        if vectors:
            if len(vectors) == 1:
                return vectors[0]
            
            weights = [1.0 - obj.metadata.distance for obj in results.objects if obj.vector.get(field_to_impute)]
            
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum > 0:
                normalized_weights = [w / weight_sum for w in weights]
                
                # Compute weighted average
                avg_vector = np.average(vectors, axis=0, weights=normalized_weights)
                
                return avg_vector.tolist()
            else:
                # Unweighted average
                avg_vector = np.mean(vectors, axis=0)
                return avg_vector.tolist()
        else:
            logger.warning(f"No vectors found for field {field_to_impute}")
            return None
            
    except Exception as e:
        logger.error(f"Error during imputation: {e}")
        return None


def get_candidates(
    person_vector: List[float],
    client: weaviate.Client,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get candidate matches for a person vector.
    
    Args:
        person_vector: Vector representation of a person
        client: Weaviate client
        limit: Maximum number of candidates
    
    Returns:
        List of candidate matches
    """
    return vector_search(
        client=client,
        query_vector=person_vector,
        field_type="person",
        limit=limit
    )


if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Indexing module loaded successfully")
