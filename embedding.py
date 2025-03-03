#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding module for the entity resolution pipeline.
Handles generation of vector embeddings for unique strings.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple

import numpy as np
import openai
from tqdm import tqdm

from utils import batch_generator, save_checkpoint

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient:
    """Client for generating embeddings using OpenAI's API."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedding client.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # Dimension for text-embedding-3-small
        
        logger.info(f"Initialized OpenAI embedding client with model: {model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        if not text:
            return [0.0] * self.dimension
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty strings
        valid_texts = [text for text in texts if text]
        valid_indices = [i for i, text in enumerate(texts) if text]
        
        if not valid_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts
            )
            
            # Reconstruct the full list with zeros for empty strings
            embeddings = [[0.0] * self.dimension] * len(texts)
            for i, embedding in zip(valid_indices, [item.embedding for item in response.data]):
                embeddings[i] = embedding
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [[0.0] * self.dimension] * len(texts)


def generate_embeddings(
    unique_strings: Dict[str, str], 
    config: Dict[str, Any],
    checkpoint_interval: int = 1000
) -> Dict[str, List[float]]:
    """
    Generate embeddings for unique strings using OpenAI's API.
    
    Args:
        unique_strings: Dictionary of hash → string value
        config: Configuration dictionary
        checkpoint_interval: Interval for saving checkpoints
    
    Returns:
        Dictionary of hash → embedding vector
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        error_msg = "OpenAI API key not found in environment variables"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Initialize embedding client
    embedding_client = OpenAIEmbeddingClient(
        api_key=api_key,
        model=config.get("embedding_model", "text-embedding-3-small")
    )
    
    # Check for existing embeddings checkpoint
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    embeddings_path = os.path.join(checkpoint_dir, "embeddings.pkl")
    
    if os.path.exists(embeddings_path) and config.get("load_checkpoints", True):
        import pickle
        with open(embeddings_path, 'rb') as f:
            existing_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(existing_embeddings)} embeddings from checkpoint")
    else:
        existing_embeddings = {}
    
    # Identify strings that need embedding
    strings_to_embed = {
        h: s for h, s in unique_strings.items() 
        if h not in existing_embeddings
    }
    logger.info(f"Generating embeddings for {len(strings_to_embed)} unique strings")
    
    # Combined embeddings dict
    embeddings = existing_embeddings.copy()
    
    # Prepare batches for processing
    hash_list = list(strings_to_embed.keys())
    batch_size = config.get("embedding_batch_size", 100)
    
    # Define a worker function for parallel processing
    def process_batch(batch_hashes):
        batch_texts = [unique_strings[h] for h in batch_hashes]
        batch_embeddings = embedding_client.generate_embeddings_batch(batch_texts)
        return {h: emb for h, emb in zip(batch_hashes, batch_embeddings)}
    
    # Use ThreadPoolExecutor for parallel processing
    num_workers = min(config.get("embedding_workers", 4), 8)  # Limit to 8 workers max
    
    if num_workers > 1:
        logger.info(f"Using {num_workers} threads for parallel embedding generation")
        batches = list(batch_generator(hash_list, batch_size))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch_results = list(tqdm(
                executor.map(process_batch, batches),
                total=len(batches),
                desc="Generating embeddings"
            ))
        
        # Combine batch results
        for batch_result in batch_results:
            embeddings.update(batch_result)
    else:
        # Sequential processing
        for i, batch_hashes in enumerate(batch_generator(hash_list, batch_size)):
            logger.info(f"Processing batch {i+1}/{len(hash_list)//batch_size + 1}")
            batch_result = process_batch(batch_hashes)
            embeddings.update(batch_result)
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint("embeddings.pkl", embeddings, config)
            
            # Rate limiting
            time.sleep(config.get("embedding_rate_limit", 0.1))
    
    # Save final checkpoint
    save_checkpoint("embeddings.pkl", embeddings, config)
    
    logger.info(f"Generated embeddings for {len(embeddings)} unique strings")
    return embeddings


if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Embedding module loaded successfully")
