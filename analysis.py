    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis and reporting module for the entity resolution pipeline.
Provides detailed insights and visualizations for each stage.
"""

import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, Optional
import csv

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from tqdm import tqdm

from utils import save_output

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisReporter:
    """Class for generating analysis reports at each pipeline stage."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get("output_dir", "data/output")
        self.report_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Create directory for figures
        self.figures_dir = os.path.join(self.report_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        logger.info(f"Initialized analysis reporter with output directory: {self.report_dir}")
    
    def analyze_preprocessing(
        self,
        unique_strings: Dict[str, str],
        string_counts: Dict[str, int],
        record_field_hashes: Dict[str, Dict[str, str]],
        field_hash_mapping: Dict[str, Dict[str, int]]
    ) -> Dict[str, Any]:
        """
        Analyze preprocessing results.
        
        Args:
            unique_strings: Dictionary of hash → string value
            string_counts: Dictionary of hash → frequency count
            record_field_hashes: Dictionary of record ID → {field → hash}
            field_hash_mapping: Dictionary of hash → {field → count}
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing preprocessing results")
        
        # Basic statistics
        total_records = len(record_field_hashes)
        total_unique_strings = len(unique_strings)
        
        # Field statistics
        field_stats = {
            "record": {"count": 0, "null_count": 0, "unique_count": 0},
            "person": {"count": 0, "null_count": 0, "unique_count": 0},
            "roles": {"count": 0, "null_count": 0, "unique_count": 0},
            "title": {"count": 0, "null_count": 0, "unique_count": 0},
            "attribution": {"count": 0, "null_count": 0, "unique_count": 0},
            "provision": {"count": 0, "null_count": 0, "unique_count": 0},
            "subjects": {"count": 0, "null_count": 0, "unique_count": 0},
            "genres": {"count": 0, "null_count": 0, "unique_count": 0},
            "relatedWork": {"count": 0, "null_count": 0, "unique_count": 0}
        }
        
        # Calculate field statistics
        field_hashes = {field: set() for field in field_stats}
        
        for record_id, fields in record_field_hashes.items():
            for field, hash_value in fields.items():
                if field in field_stats:
                    field_stats[field]["count"] += 1
                    if hash_value == "NULL":
                        field_stats[field]["null_count"] += 1
                    else:
                        field_hashes[field].add(hash_value)
        
        # Calculate unique counts
        for field, hashes in field_hashes.items():
            field_stats[field]["unique_count"] = len(hashes)
        
        # String frequency distribution
        frequency_counts = Counter(string_counts.values())
        frequency_distribution = {
            "1": frequency_counts.get(1, 0),
            "2-5": sum(frequency_counts.get(i, 0) for i in range(2, 6)),
            "6-10": sum(frequency_counts.get(i, 0) for i in range(6, 11)),
            "11-50": sum(frequency_counts.get(i, 0) for i in range(11, 51)),
            "51-100": sum(frequency_counts.get(i, 0) for i in range(51, 101)),
            "101-500": sum(frequency_counts.get(i, 0) for i in range(101, 501)),
            "501+": sum(frequency_counts.get(i, 0) for i in range(501, 10000000))
        }
        
        # Person name analysis
        person_lengths = []
        has_life_dates_count = 0
        
        for hash_value, field_counts in field_hash_mapping.items():
            if "person" in field_counts:
                person_str = unique_strings.get(hash_value, "")
                person_lengths.append(len(person_str))
                if re.search(r'\d{4}-\d{4}', person_str):
                    has_life_dates_count += 1
        
        # Create visualizations
        self._create_field_stats_chart(field_stats)
        self._create_string_frequency_chart(frequency_distribution)
        self._create_person_length_histogram(person_lengths)
        
        # Compile results
        results = {
            "total_records": total_records,
            "total_unique_strings": total_unique_strings,
            "field_statistics": field_stats,
            "string_frequency_distribution": frequency_distribution,
            "person_name_analysis": {
                "total_unique_persons": field_stats["person"]["unique_count"],
                "average_name_length": np.mean(person_lengths) if person_lengths else 0,
                "with_life_dates": has_life_dates_count,
                "without_life_dates": field_stats["person"]["unique_count"] - has_life_dates_count
            }
        }
        
        # Save report
        self._save_report("preprocessing_analysis.json", results)
        
        return results
    
    def analyze_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        field_hash_mapping: Dict[str, Dict[str, int]],
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Analyze embedding results.
        
        Args:
            embeddings: Dictionary of hash → embedding vector
            field_hash_mapping: Dictionary of hash → {field → count}
            sample_size: Number of embeddings to sample for visualization
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing embedding results")
        
        # Basic statistics
        total_embeddings = len(embeddings)
        
        # Calculate embeddings per field type
        field_embeddings = defaultdict(int)
        for hash_value, fields in field_hash_mapping.items():
            if hash_value in embeddings:
                for field in fields:
                    field_embeddings[field] += 1
        
        # Vector statistics
        embedding_lengths = []
        for vector in embeddings.values():
            if vector:
                embedding_lengths.append(np.linalg.norm(vector))
        
        # Sample embeddings for visualization
        if len(embeddings) > sample_size:
            sample_keys = np.random.choice(list(embeddings.keys()), sample_size, replace=False)
            sample_embeddings = {k: embeddings[k] for k in sample_keys}
        else:
            sample_embeddings = embeddings
        
        # Dimensionality reduction for visualization if enough samples
        if len(sample_embeddings) >= 10:
            # Create a matrix of embeddings
            embedding_matrix = np.array(list(sample_embeddings.values()))
            
            # PCA
            try:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(embedding_matrix)
                
                # Create PCA visualization
                self._create_embedding_scatter_plot(
                    pca_result, 
                    "PCA", 
                    "pca_embeddings.png"
                )
                
                # Explained variance
                explained_variance = pca.explained_variance_ratio_.sum()
            except Exception as e:
                logger.error(f"Error performing PCA: {e}")
                pca_result = None
                explained_variance = 0
            
            # t-SNE (only if enough samples)
            if len(sample_embeddings) >= 50:
                try:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(embedding_matrix)
                    
                    # Create t-SNE visualization
                    self._create_embedding_scatter_plot(
                        tsne_result,
                        "t-SNE",
                        "tsne_embeddings.png"
                    )
                except Exception as e:
                    logger.error(f"Error performing t-SNE: {e}")
                    tsne_result = None
        
        # Compile results
        results = {
            "total_embeddings": total_embeddings,
            "embeddings_by_field": dict(field_embeddings),
            "vector_statistics": {
                "min_length": min(embedding_lengths) if embedding_lengths else 0,
                "max_length": max(embedding_lengths) if embedding_lengths else 0,
                "avg_length": np.mean(embedding_lengths) if embedding_lengths else 0,
                "std_length": np.std(embedding_lengths) if embedding_lengths else 0
            },
            "pca_explained_variance": explained_variance if 'explained_variance' in locals() else 0
        }
        
        # Save report
        self._save_report("embedding_analysis.json", results)
        
        return results
    
    def analyze_indexing(
        self,
        weaviate_client: Any,
        field_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze Weaviate indexing results.
        
        Args:
            weaviate_client: Weaviate client
            field_types: List of field types
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing indexing results")
        
        if field_types is None:
            field_types = [
                "record", "person", "roles", "title", "attribution",
                "provision", "subjects", "genres", "relatedWork"
            ]
        
        # Check if client is connected
        if not weaviate_client:
            logger.error("Weaviate client not provided")
            return {"error": "Weaviate client not provided"}
        
        try:
            # Get collection statistics
            collection = weaviate_client.collections.get("UniqueStringsByField")
            
            # Get total count
            # TO FIX
            #count_result = collection.aggregate.over_all().count()
            #total_objects = count_result.total_count if hasattr(count_result, 'total_count') else 0
            #logger.info(f"UniqueStrings collection has {total_objects} objects")

            # Get counts by field type
            field_counts = {}
            try:
                from weaviate.classes.aggregate import GroupByAggregate
                # Use group_by to count by field_type
                result = collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="field_type"),
                    total_count=True
                )
                
                # Process group results
                for group in result.groups:
                    field_type = group.grouped_by
                    count = group.total_count
                    field_counts[field_type] = count
                    
                # Fill in zeros for any missing field types
                for field in field_types:
                    if field not in field_counts:
                        field_counts[field] = 0
                        
            except Exception as e:
                logger.error(f"Error getting field counts: {e}")
            
            # # Fall back to individual queries if grouping fails
            # for field in field_types:
            #     try:
            #         from weaviate.classes.query import Filter
            #         from weaviate.classes.aggregate import GroupByAggregate
            #         field_filter = Filter.by_property("field_type").equal(field)
            #         count = collection.query.fetch_objects(
            #             filters=field_filter,
            #             limit=1
            #         ).total_count                    

            #         field_counts[field] = count
            #     except Exception as e:
            #         logger.error(f"Error getting count for field {field}: {e}")
            #         field_counts[field] = 0
            
            # Create field count visualization
            self._create_field_count_chart(field_counts)
            
            # Compile results
            results = {
                #TO FIX "total_indexed_objects": total_objects,
                "objects_by_field_type": field_counts
            }
            
            # Save report
            self._save_report("indexing_analysis.json", results)
            
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing Weaviate index: {e}")
            return {"error": str(e)}
    
    def analyze_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feature engineering results.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing feature engineering results")
        
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Basic statistics
            num_samples, num_features = X.shape
            num_positive = np.sum(y)
            num_negative = num_samples - num_positive
            class_balance = num_positive / num_samples
            
            # Calculate basic stats safely
            feature_stats = {}
            for i, name in enumerate(feature_names):
                try:
                    col = X[:, i]
                    feature_stats[name] = {
                        "min": float(np.nanmin(col)),
                        "max": float(np.nanmax(col)),
                        "mean": float(np.nanmean(col)),
                        "std": float(np.nanstd(col)),
                        "median": float(np.nanmedian(col)),
                        "has_nan": bool(np.isnan(col).any()),
                        "has_inf": bool(np.isinf(col).any())
                    }
                except Exception as e:
                    logger.warning(f"Error calculating stats for feature {name}: {e}")
                    feature_stats[name] = {"error": str(e)}
            
            # Create visualizations with extra error handling
            try:
                self._create_feature_importance_chart(feature_stats)
            except Exception as e:
                logger.error(f"Error creating feature importance chart: {e}")
            
            try:
                # Use a sample of data for visualization if dataset is large
                if X.shape[0] > 10000:
                    sample_size = 10000
                    indices = np.random.choice(X.shape[0], sample_size, replace=False)
                    X_sample = X[indices]
                    y_sample = y[indices]
                    self._create_feature_distribution_chart(X_sample, y_sample, feature_names)
                else:
                    self._create_feature_distribution_chart(X, y, feature_names)
            except Exception as e:
                logger.error(f"Error creating feature distribution chart: {e}")
            
            # Compile results
            results = {
                "sample_statistics": {
                    "num_samples": int(num_samples),
                    "num_features": int(num_features),
                    "num_positive": int(num_positive),
                    "num_negative": int(num_negative),
                    "class_balance": float(class_balance)
                },
                "feature_statistics": feature_stats
            }
            
            # Save report
            self._save_report("feature_analysis.json", results)
            
            return results
        except Exception as e:
            logger.error(f"Error in feature analysis: {e}")
            return {"error": str(e)}
    
    def analyze_classification(
        self,
        classifier: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze classification results.
        
        Args:
            classifier: Trained classifier
            X_test: Test feature matrix
            y_test: Test labels
            feature_names: Names of features
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing classification results")
        
        if feature_names is None:
            feature_names = [
                'record_sim', 'person_sim', 'roles_sim', 'title_sim', 'attribution_sim',
                'provision_sim', 'subjects_sim', 'genres_sim', 'relatedWork_sim',
                'person_lev_sim', 'has_life_dates', 'temporal_overlap'
            ]
            
            # Adjust if dimensions don't match
            if len(feature_names) != X_test.shape[1]:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        # Make predictions
        y_pred_proba = classifier.predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Report
        analyze_feature_correlations_with_errors(
            X_test,
            y_test,
            y_pred,
            self.config.get("feature_names", []),
            os.path.join(self.config.get("output_dir", "data/output"), "reports")
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Misclassified examples analysis
        misclassified_indices = np.where(y_test != y_pred)[0]
        
        # Feature importance from classifier weights
        weights = classifier.weights
        feature_importance = {
            name: abs(weight)
            for name, weight in zip(feature_names, weights)
        }
        
        # Create visualizations
        self._create_confusion_matrix_plot(cm)
        self._create_roc_curve_plot(fpr, tpr, roc_auc)
        self._create_pr_curve_plot(recall_curve, precision_curve, pr_auc)
        self._create_classification_feature_importance(feature_importance)
        
        # Compile results
        results = {
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc)
            },
            "misclassification_analysis": {
                "num_misclassified": int(len(misclassified_indices)),
                "misclassification_rate": float(len(misclassified_indices) / len(y_test))
            },
            "feature_importance": feature_importance
        }
        
        # Save report
        self._save_report("classification_analysis.json", results)
        
        return results
    
    def analyze_clustering(
        self,
        entity_clusters: List[Dict[str, Any]],
        ground_truth_pairs: List[Tuple[str, str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Analyze clustering results.
        
        Args:
            entity_clusters: List of entity clusters
            ground_truth_pairs: Ground truth pairs (optional)
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing clustering results")
        
        # Basic statistics
        num_clusters = len(entity_clusters)
        
        # Size statistics
        cluster_sizes = [cluster["size"] for cluster in entity_clusters]
        
        # Confidence statistics
        confidence_values = [cluster["confidence"] for cluster in entity_clusters]
        
        # Cluster size distribution
        size_distribution = {
            "1": sum(1 for size in cluster_sizes if size == 1),
            "2-5": sum(1 for size in cluster_sizes if 2 <= size <= 5),
            "6-10": sum(1 for size in cluster_sizes if 6 <= size <= 10),
            "11-50": sum(1 for size in cluster_sizes if 11 <= size <= 50),
            "51-100": sum(1 for size in cluster_sizes if 51 <= size <= 100),
            "101+": sum(1 for size in cluster_sizes if size > 100)
        }
        
        # Confidence distribution
        confidence_distribution = {
            "0.0-0.2": sum(1 for conf in confidence_values if 0.0 <= conf < 0.2),
            "0.2-0.4": sum(1 for conf in confidence_values if 0.2 <= conf < 0.4),
            "0.4-0.6": sum(1 for conf in confidence_values if 0.4 <= conf < 0.6),
            "0.6-0.8": sum(1 for conf in confidence_values if 0.6 <= conf < 0.8),
            "0.8-1.0": sum(1 for conf in confidence_values if 0.8 <= conf <= 1.0)
        }
        
        # Create visualizations
        self._create_cluster_size_histogram(cluster_sizes)
        self._create_confidence_histogram(confidence_values)
        self._create_size_vs_confidence_scatter(cluster_sizes, confidence_values)
        
        # Compile results
        results = {
            "cluster_statistics": {
                "num_clusters": num_clusters,
                "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
                "median_cluster_size": np.median(cluster_sizes) if cluster_sizes else 0,
                "singleton_clusters": size_distribution["1"],
                "total_records_clustered": sum(cluster_sizes)
            },
            "confidence_statistics": {
                "min_confidence": min(confidence_values) if confidence_values else 0,
                "max_confidence": max(confidence_values) if confidence_values else 0,
                "avg_confidence": np.mean(confidence_values) if confidence_values else 0,
                "median_confidence": np.median(confidence_values) if confidence_values else 0
            },
            "size_distribution": size_distribution,
            "confidence_distribution": confidence_distribution
        }
        
        # Save report
        self._save_report("clustering_analysis.json", results)
        
        return results
    
    def analyze_pipeline(
        self,
        pipeline_metrics: Dict[str, Any],
        stage_timings: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze overall pipeline performance.
        
        Args:
            pipeline_metrics: Pipeline performance metrics
            stage_timings: Timing information for each stage
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing overall pipeline performance")
        
        # Calculate timing percentages
        total_time = sum(stage_timings.values())
        timing_percentages = {
            stage: (time / total_time) * 100
            for stage, time in stage_timings.items()
        }
        
        # Create timing visualization
        self._create_pipeline_timing_chart(stage_timings)
        
        # Compile results
        results = {
            "overall_metrics": pipeline_metrics,
            "timing_statistics": {
                "total_runtime": total_time,
                "stage_timings": stage_timings,
                "timing_percentages": timing_percentages
            }
        }
        
        # Save report
        self._save_report("pipeline_analysis.json", results)
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report combining all analyses.
        
        Returns:
            Comprehensive report
        """
        logger.info("Generating comprehensive report")
        
        # Load individual reports
        reports = {}
        report_files = [
            "preprocessing_analysis.json",
            "embedding_analysis.json",
            "indexing_analysis.json",
            "feature_analysis.json",
            "classification_analysis.json",
            "clustering_analysis.json",
            "pipeline_analysis.json"
        ]
        
        for file in report_files:
            try:
                with open(os.path.join(self.report_dir, file), 'r') as f:
                    reports[file.replace("_analysis.json", "")] = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Report file not found: {file}")
        
        # Compile comprehensive report
        comprehensive_report = {
            "summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "available_reports": list(reports.keys())
            },
            "reports": reports
        }
        
        # Save comprehensive report
        self._save_report("comprehensive_report.json", comprehensive_report)
        
        return comprehensive_report
    
    def _save_report(self, filename: str, data: Dict[str, Any]) -> None:
        """
        Save report to a file.
        
        Args:
            filename: Report filename
            data: Report data
        """
        file_path = os.path.join(self.report_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved report: {file_path}")
    
    def _create_field_stats_chart(self, field_stats: Dict[str, Dict[str, int]]) -> None:
        """
        Create field statistics chart.
        
        Args:
            field_stats: Field statistics
        """
        # Prepare data
        fields = list(field_stats.keys())
        total_counts = [field_stats[field]["count"] for field in fields]
        null_counts = [field_stats[field]["null_count"] for field in fields]
        unique_counts = [field_stats[field]["unique_count"] for field in fields]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar chart
        x = np.arange(len(fields))
        width = 0.25
        
        plt.bar(x - width, total_counts, width, label='Total')
        plt.bar(x, null_counts, width, label='Null')
        plt.bar(x + width, unique_counts, width, label='Unique')
        
        plt.xlabel('Fields')
        plt.ylabel('Count')
        plt.title('Field Statistics')
        plt.xticks(x, fields, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "field_statistics.png"))
        plt.close()
    
    def _create_string_frequency_chart(self, frequency_distribution: Dict[str, int]) -> None:
        """
        Create string frequency chart.
        
        Args:
            frequency_distribution: Frequency distribution
        """
        # Prepare data
        categories = list(frequency_distribution.keys())
        counts = list(frequency_distribution.values())
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        plt.bar(categories, counts)
        
        plt.xlabel('Frequency Range')
        plt.ylabel('Count')
        plt.title('String Frequency Distribution')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "string_frequency.png"))
        plt.close()
    
    def _create_person_length_histogram(self, person_lengths: List[int]) -> None:
        """
        Create person name length histogram.
        
        Args:
            person_lengths: List of person name lengths
        """
        if not person_lengths:
            logger.warning("No person lengths provided")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(person_lengths, bins=30, edgecolor='black')
        
        plt.xlabel('Name Length')
        plt.ylabel('Count')
        plt.title('Person Name Length Distribution')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "person_length_histogram.png"))
        plt.close()
    
    def _create_embedding_scatter_plot(
        self,
        embedding_result: np.ndarray,
        method: str,
        filename: str
    ) -> None:
        """
        Create embedding scatter plot.
        
        Args:
            embedding_result: Dimensionality reduction result
            method: Dimensionality reduction method
            filename: Output filename
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(embedding_result[:, 0], embedding_result[:, 1], alpha=0.5)
        
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.title(f'{method} Visualization of Embeddings')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, filename))
        plt.close()
    
    def _create_field_count_chart(self, field_counts: Dict[str, int]) -> None:
        """
        Create field count chart.
        
        Args:
            field_counts: Field counts
        """
        # Prepare data
        fields = list(field_counts.keys())
        counts = list(field_counts.values())
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        plt.bar(fields, counts)
        
        plt.xlabel('Field Type')
        plt.ylabel('Count')
        plt.title('Objects by Field Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "field_counts.png"))
        plt.close()
    
    def _create_feature_importance_chart(self, feature_importance: Dict[str, float]) -> None:
        """
        Create feature importance chart.
        
        Args:
            feature_importance: Feature importance scores
        """
        try:
            # Prepare data - make sure we're sorting by value, not by dict
            if isinstance(next(iter(feature_importance.values()), None), dict):
                # Handle case where values are dictionaries
                sorted_features = sorted(
                    [(name, values.get("mean", 0)) for name, values in feature_importance.items()],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            else:
                # Handle case where values are directly comparable
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            
            features = [item[0] for item in sorted_features]
            importance = [item[1] for item in sorted_features]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create bar chart
            plt.bar(features, importance)
            
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.figures_dir, "feature_importance.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            # Create simple error message figure
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating feature importance chart:\n{str(e)}", 
                    ha='center', va='center', fontsize=12)
            plt.savefig(os.path.join(self.figures_dir, "feature_importance_error.png"))
            plt.close()
    
    def _create_feature_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """
        Create feature correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            feature_names: Feature names
        """
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            xticklabels=feature_names,
            yticklabels=feature_names,
            vmin=-1,
            vmax=1
        )
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "feature_correlation.png"))
        plt.close()
    
    def _create_feature_distribution_chart(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        max_features: int = 6
    ) -> None:
        """
        Create feature distribution chart.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Feature names
            max_features: Maximum number of features to display
        """
        try:
            # Safety check for NaN or inf values
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning("Feature matrix contains NaN or infinite values. Cleaning data for visualization.")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
                
            # Limit size of data to avoid memory issues
            max_samples = 10000
            if X.shape[0] > max_samples:
                logger.info(f"Sampling {max_samples} rows for visualization (from {X.shape[0]} total)")
                indices = np.random.choice(X.shape[0], max_samples, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # Limit the number of features to display
            if len(feature_names) > max_features:
                # Select most important features based on class separation
                feature_importance = []
                for i in range(X_sample.shape[1]):
                    if i < len(feature_names):  # Ensure we don't go out of bounds
                        try:
                            pos_mean = np.mean(X_sample[y_sample == 1, i])
                            neg_mean = np.mean(X_sample[y_sample == 0, i])
                            importance = abs(pos_mean - neg_mean)
                            feature_importance.append((i, importance))
                        except:
                            feature_importance.append((i, 0.0))
                            
                # Sort by importance
                top_indices = [idx for idx, _ in sorted(feature_importance, key=lambda x: x[1], reverse=True)[:max_features]]
                selected_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_indices]
                selected_indices = top_indices
            else:
                selected_features = feature_names
                selected_indices = list(range(min(len(feature_names), X_sample.shape[1])))
            
            # Create figure
            fig, axes = plt.subplots(nrows=len(selected_indices), figsize=(10, 3*len(selected_indices)))
            
            if len(selected_indices) == 1:
                axes = [axes]
            
            # Create distribution plots
            for i, (feature_idx, ax) in enumerate(zip(selected_indices, axes)):
                try:
                    if feature_idx < X_sample.shape[1]:  # Safety check
                        feature_values_pos = X_sample[y_sample == 1, feature_idx]
                        feature_values_neg = X_sample[y_sample == 0, feature_idx]
                        
                        # Use histogram with fixed number of bins to avoid memory issues
                        ax.hist(feature_values_pos, bins=20, alpha=0.5, label='Match', color='blue')
                        ax.hist(feature_values_neg, bins=20, alpha=0.5, label='Non-match', color='red')
                        
                        # Avoid seaborn histplot which can cause memory issues
                        # sns.histplot(feature_values_pos, ax=ax, color='blue', alpha=0.5, label='Match', bins=20)
                        # sns.histplot(feature_values_neg, ax=ax, color='red', alpha=0.5, label='Non-match', bins=20)
                        
                        if i < len(selected_features):
                            ax.set_title(f'Distribution of {selected_features[i]}')
                        else:
                            ax.set_title(f'Distribution of Feature {feature_idx}')
                        
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Count')
                        ax.legend()
                except Exception as e:
                    logger.error(f"Error plotting feature {i}: {e}")
                    ax.text(0.5, 0.5, f"Error plotting this feature: {str(e)[:50]}...", 
                        ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.figures_dir, "feature_distributions.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating feature distribution chart: {e}")
            # Create simple error message figure
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating feature distributions:\n{str(e)}", 
                    ha='center', va='center', fontsize=12)
            plt.savefig(os.path.join(self.figures_dir, "feature_distributions_error.png"))
            plt.close()
    
    def _create_confusion_matrix_plot(self, cm: np.ndarray) -> None:
        """
        Create confusion matrix plot.
        
        Args:
            cm: Confusion matrix
        """
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Non-match', 'Match'],
            yticklabels=['Non-match', 'Match']
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "confusion_matrix.png"))
        plt.close()
    
    def _create_roc_curve_plot(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float
    ) -> None:
        """
        Create ROC curve plot.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: ROC AUC score
        """
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Create ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "roc_curve.png"))
        plt.close()
    
    def _create_pr_curve_plot(
        self,
        recall: np.ndarray,
        precision: np.ndarray,
        pr_auc: float
    ) -> None:
        """
        Create precision-recall curve plot.
        
        Args:
            recall: Recall values
            precision: Precision values
            pr_auc: Precision-recall AUC
        """
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Create precision-recall curve
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "pr_curve.png"))
        plt.close()
    
    def _create_classification_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """
        Create classifier feature importance chart.
        
        Args:
            feature_importance: Feature importance scores
        """
        # Prepare data
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        features = [item[0] for item in sorted_features]
        importance = [item[1] for item in sorted_features]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        plt.bar(features, importance)
        
        plt.xlabel('Feature')
        plt.ylabel('Weight Magnitude')
        plt.title('Classifier Feature Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "classifier_feature_importance.png"))
        plt.close()
    
    def _create_cluster_size_histogram(self, cluster_sizes: List[int]) -> None:
        """
        Create cluster size histogram.
        
        Args:
            cluster_sizes: Cluster sizes
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(cluster_sizes, bins=30, edgecolor='black')
        
        plt.xlabel('Cluster Size')
        plt.ylabel('Count')
        plt.title('Cluster Size Distribution')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "cluster_size_histogram.png"))
        plt.close()
    
    def _create_confidence_histogram(self, confidence_values: List[float]) -> None:
        """
        Create confidence histogram.
        
        Args:
            confidence_values: Confidence values
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(confidence_values, bins=20, range=(0, 1), edgecolor='black')
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Cluster Confidence Distribution')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "confidence_histogram.png"))
        plt.close()
    
    def _create_size_vs_confidence_scatter(
        self,
        sizes: List[int],
        confidences: List[float]
    ) -> None:
        """
        Create size vs. confidence scatter plot.
        
        Args:
            sizes: Cluster sizes
            confidences: Confidence values
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(sizes, confidences, alpha=0.5)
        
        plt.xlabel('Cluster Size')
        plt.ylabel('Confidence')
        plt.title('Cluster Size vs. Confidence')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "size_vs_confidence.png"))
        plt.close()
    
    def _create_pipeline_timing_chart(self, stage_timings: Dict[str, float]) -> None:
        """
        Create pipeline timing chart.
        
        Args:
            stage_timings: Stage timing information
        """
        # Prepare data
        stages = list(stage_timings.keys())
        times = list(stage_timings.values())
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        plt.bar(stages, times)
        
        plt.xlabel('Pipeline Stage')
        plt.ylabel('Time (seconds)')
        plt.title('Pipeline Stage Timing')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.figures_dir, "pipeline_timing.png"))
        plt.close()

def analyze_feature_correlations_with_errors(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    output_dir: str
) -> None:
    """
    Analyze correlations between features and prediction errors.
    
    Args:
        X_test: Test feature matrix
        y_test: True labels
        y_pred: Predicted labels
        feature_names: Names of features
        output_dir: Directory to save analysis
    """
    logger.info("Analyzing feature correlations with prediction errors")
    
    # Create error indicator (1 if misclassified, 0 if correct)
    errors = (y_test != y_pred).astype(int)
    
    # Calculate correlation between each feature and errors
    correlations = []
    for i in range(X_test.shape[1]):
        if i < len(feature_names):
            try:
                corr = np.corrcoef(X_test[:, i], errors)[0, 1]
                correlations.append((feature_names[i], corr))
            except:
                # Handle potential errors (e.g., constant feature values)
                correlations.append((feature_names[i], 0.0))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_error_correlations.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Correlation with Errors'])
        
        for feature, corr in correlations:
            writer.writerow([feature, corr])
    
    logger.info(f"Saved feature error correlation analysis to {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot top 20 features or all if less than 20
    top_n = min(20, len(correlations))
    top_features = [x[0] for x in correlations[:top_n]]
    top_correlations = [x[1] for x in correlations[:top_n]]
    
    plt.barh(range(top_n), [abs(c) for c in top_correlations], color=['red' if c > 0 else 'blue' for c in top_correlations])
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Absolute Correlation with Prediction Errors')
    plt.title('Features Most Correlated with Prediction Errors')
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(output_dir, "figures", "feature_error_correlations.png")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path)
    plt.close()

def analyze_pipeline_performance(
    pipeline_instance: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze pipeline performance and generate reports.
    
    Args:
        pipeline_instance: Pipeline instance
        config: Configuration dictionary
        
    Returns:
        Analysis results
    """
    logger.info("Analyzing pipeline performance")
    
    # Initialize reporter
    reporter = AnalysisReporter(config)
    
    # Load checkpoints
    unique_strings = pipeline_instance.load_checkpoint("unique_strings.json")
    string_counts = pipeline_instance.load_checkpoint("string_counts.json")
    record_field_hashes = pipeline_instance.load_checkpoint("record_field_hashes.json")
    field_hash_mapping = pipeline_instance.load_checkpoint("field_hash_mapping.json")
    embeddings = pipeline_instance.load_checkpoint("embeddings.pkl")
    classifier = pipeline_instance.load_checkpoint("classifier.pkl")
    
    # Load ground truth
    ground_truth_pairs = []
    try:
        from preprocessing import load_ground_truth
        ground_truth_pairs = load_ground_truth(config["ground_truth_path"])
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
    
    # Initialize stage reports
    reports = {}
    
    # Analyze preprocessing
    if all([unique_strings, string_counts, record_field_hashes, field_hash_mapping]):
        reports["preprocessing"] = reporter.analyze_preprocessing(
            unique_strings, string_counts, record_field_hashes, field_hash_mapping
        )
    
    # Analyze embeddings
    if embeddings and field_hash_mapping:
        reports["embedding"] = reporter.analyze_embeddings(
            embeddings, field_hash_mapping
        )
    
    # Connect to Weaviate for indexing analysis
    try:
        from indexing import connect_to_weaviate
        weaviate_client = connect_to_weaviate(config)
        reports["indexing"] = reporter.analyze_indexing(weaviate_client)
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
    
    # Load entity clusters
    entity_clusters = []
    try:
        output_dir = config.get("output_dir", "data/output")
        clusters_path = os.path.join(output_dir, "entity_clusters.jsonl")
        
        if os.path.exists(clusters_path):
            with open(clusters_path, 'r') as f:
                entity_clusters = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error loading entity clusters: {e}")
    
    # Analyze clustering
    if entity_clusters:
        reports["clustering"] = reporter.analyze_clustering(
            entity_clusters, ground_truth_pairs
        )
    
    # Generate comprehensive report
    comprehensive_report = reporter.generate_comprehensive_report()
    
    return comprehensive_report

def analyze_test_results(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: Any,
    test_pairs: List[Tuple[str, str, bool]],
    record_field_hashes: Dict[str, Dict[str, str]],
    unique_strings: Dict[str, str],
    feature_names: List[str],
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Complete analysis of test results with detailed reports.
    
    Args:
        X_test: Test feature matrix
        y_test: Test labels
        classifier: Trained classifier
        test_pairs: Original test record pairs
        record_field_hashes: Record field hashes
        unique_strings: String lookup dictionary
        feature_names: Names of features
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    import csv
    import os
    import json
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    logger.info("Analyzing test results and generating detailed reports")
    
    # Create output directory
    output_dir = config.get("output_dir", "data/output")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Ensure feature names match feature dimensions
    if X_test.shape[1] != len(feature_names):
        logger.warning(f"Feature dimension mismatch: X_test has {X_test.shape[1]} features but {len(feature_names)} names provided")
        
        # Create descriptive feature names if possible, otherwise use generic names
        try:
            # Try to extend with base names we know
            base_names = [
                'person_sim', 'record_sim', 'title_sim', 'roles_sim', 'attribution_sim', 
                'provision_sim', 'subjects_sim', 'genres_sim', 'relatedWork_sim',
                'person_lev_sim', 'has_life_dates', 'temporal_overlap'
            ]
            
            # If we have more features than names, extend with descriptive names if possible
            if len(feature_names) < X_test.shape[1]:
                extended_names = feature_names.copy()
                
                # These are common feature names in enhanced feature vectors
                enhanced_names = [
                    'birth_year_similarity', 'death_year_similarity', 'active_period_similarity',
                    'provision_year_overlap', 'date_range_overlap', 'temporal_compatibility',
                    'is_author_consistency', 'is_subject_consistency', 'is_editor_consistency',
                    'is_translator_consistency', 'is_contributor_consistency',
                    'person_sim_person_lev_sim_interaction', 'temporal_overlap_person_sim_interaction',
                    'exact_name_temporal_match', 'person_sim_squared', 'person_title_harmonic'
                ]
                
                # Add common enhanced names if needed
                for name in enhanced_names:
                    if len(extended_names) < X_test.shape[1] and name not in extended_names:
                        extended_names.append(name)
                
                # If still not enough, add generic names
                while len(extended_names) < X_test.shape[1]:
                    extended_names.append(f"feature_{len(extended_names)}")
                
                feature_names = extended_names
        except Exception as e:
            logger.error(f"Error creating descriptive feature names: {e}")
            # Fall back to generic names
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    # Log the feature names being used
    logger.info(f"Using {len(feature_names)} feature names for analysis")
    if len(feature_names) > 10:
        logger.info(f"First 10 feature names: {feature_names[:10]}...")
    else:
        logger.info(f"Feature names: {feature_names}")
    
    # Generate predictions
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    try:
        accuracy = np.mean(y_pred == y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "test_samples": len(y_test),
            "positive_samples": int(np.sum(y_test)),
            "predicted_positive": int(np.sum(y_pred)),
            "true_positives": int(np.sum((y_pred == 1) & (y_test == 1))),
            "false_positives": int(np.sum((y_pred == 1) & (y_test == 0))),
            "true_negatives": int(np.sum((y_pred == 0) & (y_test == 0))),
            "false_negatives": int(np.sum((y_pred == 0) & (y_test == 1)))
        }
        
        logger.info(f"Test set evaluation:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        metrics = {"error": str(e)}
    
    # 1. Export all test data with predictions and features
    try:
        logger.info("Exporting complete test dataset to CSV")
        csv_path = os.path.join(reports_dir, "test_dataset_complete.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = [
                'record_id1', 'record_id2', 
                'person1', 'person2',
                'true_label', 'predicted_label', 'predicted_prob', 'is_correct'
            ]
            headers.extend(feature_names)
            writer.writerow(headers)
            
            for idx in range(min(len(test_pairs), len(y_test))):
                id1, id2, _ = test_pairs[idx]
                
                # Get person names
                fields1 = record_field_hashes.get(id1, {})
                fields2 = record_field_hashes.get(id2, {})
                person1 = unique_strings.get(fields1.get('person', "NULL"), "")
                person2 = unique_strings.get(fields2.get('person', "NULL"), "")
                
                # Create row
                row = [
                    id1, id2,
                    person1, person2,
                    int(y_test[idx]), 
                    int(y_pred[idx]),
                    float(y_pred_proba[idx]),
                    int(y_test[idx] == y_pred[idx])
                ]
                
                # Add feature values
                row.extend(X_test[idx].tolist())
                writer.writerow(row)
                
        logger.info(f"Saved complete test dataset to {csv_path}")
    except Exception as e:
        logger.error(f"Error exporting test dataset: {e}")
    
    # 2. Export misclassified pairs with detailed information
    try:
        logger.info("Generating report of misclassified pairs")
        misclassified_indices = np.where(y_test != y_pred)[0]
        
        if len(misclassified_indices) > 0:
            csv_path = os.path.join(reports_dir, "misclassified_pairs.csv")
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                headers = [
                    'record_id1', 'record_id2', 
                    'person1', 'person2',
                    'title1', 'title2',
                    'provision1', 'provision2',
                    'true_label', 'predicted_label', 'predicted_prob'
                ]
                headers.extend(feature_names)
                writer.writerow(headers)
                
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
                            int(y_test[idx]), 
                            int(y_pred[idx]),
                            float(y_pred_proba[idx])
                        ]
                        
                        # Add feature values
                        row.extend(X_test[idx].tolist())
                        writer.writerow(row)
                
            logger.info(f"Saved report of {len(misclassified_indices)} misclassified pairs to {csv_path}")
        else:
            logger.info("No misclassified pairs found")
    except Exception as e:
        logger.error(f"Error generating misclassified pairs report: {e}")
    
    # 3. Feature correlation with errors analysis
    try:
        logger.info("Analyzing feature correlations with errors")
        errors = (y_test != y_pred).astype(int)
        
        # Calculate correlation between each feature and errors
        correlations = []
        for i in range(X_test.shape[1]):
            if i < len(feature_names):
                try:
                    corr = np.corrcoef(X_test[:, i], errors)[0, 1]
                    correlations.append((feature_names[i], corr))
                except:
                    correlations.append((feature_names[i], 0.0))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Save to CSV
        csv_path = os.path.join(reports_dir, "feature_error_correlations.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Feature', 'Correlation with Errors'])
            
            for feature, corr in correlations:
                writer.writerow([feature, corr])
        
        logger.info(f"Saved feature error correlation analysis to {csv_path}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 features or all if less than 20
        top_n = min(20, len(correlations))
        top_features = [x[0] for x in correlations[:top_n]]
        top_correlations = [x[1] for x in correlations[:top_n]]
        
        plt.barh(range(top_n), [abs(c) for c in top_correlations], color=['red' if c > 0 else 'blue' for c in top_correlations])
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Absolute Correlation with Prediction Errors')
        plt.title('Features Most Correlated with Prediction Errors')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(figures_dir, "feature_error_correlations.png"))
        plt.close()
    except Exception as e:
        logger.error(f"Error analyzing feature correlations: {e}")
    
    # Save metrics
    try:
        metrics_path = os.path.join(reports_dir, "test_evaluation_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved test evaluation metrics to {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
    
    return metrics


if __name__ == "__main__":
    # Simple test to ensure the module loads correctly
    print("Analysis module loaded successfully")
