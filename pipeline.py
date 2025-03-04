#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline module for the entity resolution system.
Orchestrates the entire entity resolution process.
"""

import argparse
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Any, Optional

from analysis import AnalysisReporter, analyze_pipeline_performance, analyze_test_results

from preprocessing import (
    preprocess_catalog_data, get_input_files, 
    load_ground_truth, setup_directories
)
from embedding import generate_embeddings
from indexing import index_embeddings, connect_to_weaviate
from classification import (
    engineer_features, train_classifier,
    classify_and_cluster, evaluate_clusters
)
from utils import load_config, setup_logger, save_output


class EntityResolutionPipeline:
    """
    Main pipeline class for entity resolution.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path or "config.json")
        
        # Set up logging
        log_dir = self.config.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"pipeline_{time.strftime('%Y%m%d-%H%M%S')}.log")
        self.logger = setup_logger(
            log_level=self.config.get("log_level", "INFO"),
            log_file=log_file
        )
        
        # Create necessary directories
        setup_directories(self.config)
        
        self.logger.info("Initialized entity resolution pipeline")
        self.logger.info(f"Configuration: {self.config}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete entity resolution pipeline.
        
        Returns:
            Evaluation metrics
        """
        start_time = time.time()
        self.logger.info("Starting full pipeline")
        
        # Initialize timing tracker
        stage_timings = {}
        
        # Initialize analysis reporter
        reporter = AnalysisReporter(self.config)
        
        # Initialize the Weaviate client as None
        weaviate_client = None

        try:
            # Step 1: Preprocessing
            self.logger.info("Step 1: Preprocessing data")
            preprocessing_start = time.time()
            
            input_files = get_input_files(self.config["input_dir"])
            
            unique_strings, string_counts, record_field_hashes, field_hash_mapping = preprocess_catalog_data(
                input_files, 
                self.config,
                dev_mode=self.config.get("dev_mode", False)
            )
            
            stage_timings["preprocessing"] = time.time() - preprocessing_start
            
            # Analyze preprocessing results
            if self.config.get("generate_analysis", True):
                reporter.analyze_preprocessing(
                    unique_strings, string_counts, record_field_hashes, field_hash_mapping
                )
            
            # Step 2: Vector Embedding
            self.logger.info("Step 2: Generating embeddings")
            embedding_start = time.time()
            
            embeddings = generate_embeddings(unique_strings, self.config)
            
            stage_timings["embedding"] = time.time() - embedding_start
            
            # Analyze embedding results
            if self.config.get("generate_analysis", True):
                reporter.analyze_embeddings(embeddings, field_hash_mapping)
            
            # Step 3: Weaviate Indexing
            self.logger.info("Step 3: Indexing embeddings in Weaviate")
            indexing_start = time.time()
            
            weaviate_client = index_embeddings(
                unique_strings, 
                embeddings, 
                field_hash_mapping,
                string_counts,
                self.config
            )
            
            stage_timings["indexing"] = time.time() - indexing_start
            
            # Analyze indexing results
            if self.config.get("generate_analysis", True):
                reporter.analyze_indexing(weaviate_client)
            
            # Step 4: Feature Engineering & Training
            self.logger.info("Step 4: Engineering features and training classifier")
            feature_start = time.time()

            # Load ground truth pairs
            ground_truth_pairs = load_ground_truth(self.config["ground_truth_path"])

            # Split ground truth pairs BEFORE feature engineering
            from sklearn.model_selection import train_test_split
            train_pairs, test_pairs = train_test_split(
                ground_truth_pairs,
                test_size=1-self.config.get("train_test_split", 0.8),
                random_state=self.config.get("random_seed", 42)
            )

            self.logger.info(f"Split into {len(train_pairs)} training pairs and {len(test_pairs)} test pairs")

            # Only engineer features for training data
            X_train, y_train, train_feature_names = engineer_features(
                train_pairs,  # Only use training pairs here!
                record_field_hashes, 
                unique_strings, 
                embeddings, 
                weaviate_client, 
                self.config
            )

            # Store feature names in config
            self.config["feature_names"] = train_feature_names

            # Train classifier
            classifier = train_classifier(X_train, y_train, self.config)

            # Analyze feature engineering results
            if self.config.get("generate_analysis", True):
                reporter.analyze_features(X_train, y_train)

            # Generate features for test set for evaluation
            self.logger.info("Engineering features for test set...")
            X_test, y_test, test_feature_names = engineer_features(
                test_pairs,
                record_field_hashes,
                unique_strings,
                embeddings,
                weaviate_client,
                self.config
            )

            # Report results
            from classification import generate_detailed_reports
            # Use the actual feature names from test data
            # Call the new function instead of generate_detailed_reports
            test_metrics = analyze_test_results(
                X_test,
                y_test,
                classifier,
                test_pairs,
                record_field_hashes,
                unique_strings,
                self.config.get("feature_names", []),
                self.config
            )            

            stage_timings["feature_engineering_training"] = time.time() - feature_start

            # Analyze classification results
            if self.config.get("generate_analysis", True):
                reporter.analyze_classification(classifier, X_test, y_test)
            
            # Step 5: Classification & Clustering
            self.logger.info("Step 5: Classifying and clustering entities")
            clustering_start = time.time()
            
            entity_clusters = classify_and_cluster(
                record_field_hashes, 
                unique_strings, 
                embeddings,
                classifier, 
                weaviate_client, 
                self.config
            )
            
            stage_timings["classification_clustering"] = time.time() - clustering_start
            
            # Analyze clustering results
            if self.config.get("generate_analysis", True):
                reporter.analyze_clustering(entity_clusters, ground_truth_pairs)
            
            # Step 6: Evaluation
            self.logger.info("Step 6: Evaluating results")
            evaluation_start = time.time()
            
            evaluation_results = evaluate_clusters(
                entity_clusters, 
                ground_truth_pairs, 
                self.config
            )
            
            stage_timings["evaluation"] = time.time() - evaluation_start
            
            # Log timing
            end_time = time.time()
            elapsed = end_time - start_time
            self.logger.info(f"Pipeline completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            
            # Generate pipeline analysis
            if self.config.get("generate_analysis", True):
                reporter.analyze_pipeline(evaluation_results, stage_timings)
                comprehensive_report = reporter.generate_comprehensive_report()
                self.logger.info(f"Generated comprehensive analysis report")
            
            return evaluation_results
        
        finally:
            # Clean up and close Weaviate client connection
            if weaviate_client is not None:
                try:
                    weaviate_client.close()
                    self.logger.info("Weaviate client connection closed")
                except Exception as e:
                    self.logger.warning(f"Error closing Weaviate client: {e}")
    
    def run_module(self, module_name: str, **kwargs) -> Any:
        """
        Run a specific module with optional input data.
        
        Args:
            module_name: Name of the module to run
            **kwargs: Additional arguments for the module
            
        Returns:
            Module output
        """
        self.logger.info(f"Running module: {module_name}")
        
        if module_name == "preprocessing":
            input_files = get_input_files(self.config["input_dir"])
            return preprocess_catalog_data(
                input_files, 
                self.config,
                dev_mode=kwargs.get("dev_mode", self.config.get("dev_mode", False))
            )
        
        elif module_name == "embedding":
            unique_strings = kwargs.get("unique_strings")
            if not unique_strings:
                checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
                unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.json")
                
                if not os.path.exists(unique_strings_path):
                    self.logger.error("No unique_strings provided and no checkpoint found")
                    return None
                
                import json
                with open(unique_strings_path, 'r') as f:
                    unique_strings = json.load(f)
                
            return generate_embeddings(unique_strings, self.config)
        
        elif module_name == "indexing":
            unique_strings = kwargs.get("unique_strings")
            embeddings = kwargs.get("embeddings")
            field_hash_mapping = kwargs.get("field_hash_mapping")
            string_counts = kwargs.get("string_counts")
            
            if not all([unique_strings, embeddings, field_hash_mapping, string_counts]):
                self.logger.error("Missing required inputs for indexing module")
                return None
            
            return index_embeddings(
                unique_strings, 
                embeddings, 
                field_hash_mapping,
                string_counts,
                self.config
            )
        
        elif module_name == "feature_engineering":
            record_field_hashes = kwargs.get("record_field_hashes")
            unique_strings = kwargs.get("unique_strings")
            embeddings = kwargs.get("embeddings")
            weaviate_client = kwargs.get("weaviate_client")
            ground_truth_pairs = kwargs.get("ground_truth_pairs")
            
            if not ground_truth_pairs:
                ground_truth_pairs = load_ground_truth(self.config["ground_truth_path"])
            
            if not weaviate_client:
                weaviate_client = connect_to_weaviate(self.config)
            
            if not all([record_field_hashes, unique_strings, embeddings]):
                self.logger.error("Missing required inputs for feature engineering module")
                return None
            
            return engineer_features(
                ground_truth_pairs, 
                record_field_hashes, 
                unique_strings, 
                embeddings, 
                weaviate_client, 
                self.config
            )
        
        elif module_name == "training":
            X = kwargs.get("X")
            y = kwargs.get("y")
            
            if not all([X, y]):
                self.logger.error("Missing required inputs for training module")
                return None
            
            return train_classifier(X, y, self.config)
        
        elif module_name == "classification":
            record_field_hashes = kwargs.get("record_field_hashes")
            unique_strings = kwargs.get("unique_strings")
            embeddings = kwargs.get("embeddings")
            classifier = kwargs.get("classifier")
            weaviate_client = kwargs.get("weaviate_client")
            
            if not weaviate_client:
                weaviate_client = connect_to_weaviate(self.config)
            
            if not all([record_field_hashes, unique_strings, embeddings, classifier]):
                self.logger.error("Missing required inputs for classification module")
                return None
            
            return classify_and_cluster(
                record_field_hashes, 
                unique_strings, 
                embeddings,
                classifier, 
                weaviate_client, 
                self.config
            )
        
        elif module_name == "evaluation":
            entity_clusters = kwargs.get("entity_clusters")
            ground_truth_pairs = kwargs.get("ground_truth_pairs")
            
            if not ground_truth_pairs:
                ground_truth_pairs = load_ground_truth(self.config["ground_truth_path"])
            
            if not entity_clusters:
                self.logger.error("Missing required inputs for evaluation module")
                return None
            
            return evaluate_clusters(
                entity_clusters, 
                ground_truth_pairs, 
                self.config
            )
        
        else:
            self.logger.error(f"Unknown module: {module_name}")
            return None
    
    def load_checkpoint(self, name: str) -> Any:
        """
        Load data from a checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Loaded data
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
        path = os.path.join(checkpoint_dir, name)
        
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint not found: {path}")
            return None
        
        try:
            if path.endswith('.json'):
                import json
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.endswith('.pkl'):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                self.logger.error(f"Unknown checkpoint format: {path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {path}: {e}")
            return None


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--config", help="Path to configuration file", default="config.json")
    parser.add_argument("--module", help="Run a specific module", choices=[
        "preprocessing", "embedding", "indexing", 
        "feature_engineering", "training", "classification", 
        "evaluation", "analysis", "full"
    ])
    parser.add_argument("--dev", help="Run in development mode", action="store_true")
    parser.add_argument("--no-analysis", help="Skip analysis generation", action="store_true")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EntityResolutionPipeline(args.config)
    
    # Override config options if specified
    if args.dev:
        pipeline.config["dev_mode"] = True
        
    if args.no_analysis:
        pipeline.config["generate_analysis"] = False
    
    # Run specified module or full pipeline
    if args.module and args.module != "full":
        if args.module == "analysis":
            # Run analysis on existing data
            result = analyze_pipeline_performance(pipeline, pipeline.config)
            print("Analysis module completed")
        else:
            result = pipeline.run_module(args.module)
            print(f"Module {args.module} completed")
    else:
        result = pipeline.run_full_pipeline()
        print("Full pipeline completed")
        print(f"Evaluation results: {result}")


if __name__ == "__main__":
    main()
