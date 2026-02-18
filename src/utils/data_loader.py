"""
Data Loading and Preprocessing Utilities for Protein Binder Analysis

This module handles loading, parsing, and preprocessing of protein binder datasets
with experimental and computational evaluations.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ProteinDataLoader:
    """Load and preprocess protein binder data from CSV files."""

    def __init__(self, filepath: str):
        """
        Initialize the data loader.

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing protein data
        """
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Returns
        -------
        pd.DataFrame
            Raw protein data
        """
        print(f"Loading data from {self.filepath}...")
        self.raw_data = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.raw_data)} protein designs")
        return self.raw_data

    def parse_evaluations(self, evaluations_str: str) -> List[Dict]:
        """
        Parse the evaluations JSON string into a list of dictionaries.

        Parameters
        ----------
        evaluations_str : str
            JSON string containing evaluations

        Returns
        -------
        List[Dict]
            Parsed evaluations
        """
        try:
            if pd.isna(evaluations_str) or evaluations_str == '[]':
                return []
            return json.loads(evaluations_str)
        except (json.JSONDecodeError, TypeError):
            return []

    def extract_metrics_from_evaluations(self, evaluations: List[Dict],
                                          filter_type: Optional[str] = None) -> Dict:
        """
        Extract specific metrics from evaluations list.

        Parameters
        ----------
        evaluations : List[Dict]
            List of evaluation dictionaries
        filter_type : str, optional
            Filter to only 'computational' or 'experimental' features.
            If None, extracts all features.

        Returns
        -------
        Dict
            Dictionary of extracted metrics
        """
        metrics = {}

        # Extract various metrics
        for eval_item in evaluations:
            metric_name = eval_item.get('metric', '')
            value = eval_item.get('value')
            eval_type = eval_item.get('type', '')
            target = eval_item.get('target', 'unknown')

            # Filter by type if specified
            if filter_type is not None and eval_type != filter_type:
                continue

            # Create unique key for target-specific metrics
            if target and target != 'unknown':
                key = f"{metric_name}_{target}_{eval_type}"
            else:
                key = f"{metric_name}_{eval_type}"

            # Handle different value types
            if isinstance(value, dict):
                # For nested JSON values, extract relevant fields
                if 'value' in value:
                    metrics[key] = value['value']
                elif 'metrics' in value:
                    # Handle domain match metrics
                    for sub_metric in value['metrics']:
                        sub_key = f"{metric_name}_{sub_metric.get('slug', '')}_{eval_type}"
                        metrics[sub_key] = sub_metric.get('value')
            else:
                metrics[key] = value

        return metrics

    def process_data(self) -> pd.DataFrame:
        """
        Process raw data and extract features from evaluations.

        Returns
        -------
        pd.DataFrame
            Processed data with extracted features
        """
        if self.raw_data is None:
            self.load_data()

        print("Processing evaluations and extracting features...")

        # Parse evaluations for each row
        parsed_evaluations = self.raw_data['evaluations'].apply(self.parse_evaluations)

        # Extract metrics from evaluations
        metrics_list = parsed_evaluations.apply(self.extract_metrics_from_evaluations)

        # Convert list of dicts to DataFrame
        metrics_df = pd.DataFrame(metrics_list.tolist())

        # Combine with original data
        self.processed_data = pd.concat([
            self.raw_data.drop('evaluations', axis=1),
            metrics_df
        ], axis=1)

        # Add sequence-based features
        self.processed_data['sequence_length'] = self.raw_data['sequence'].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )

        print(f"Extracted {len(metrics_df.columns)} features from evaluations")
        print(f"Total features: {len(self.processed_data.columns)}")

        return self.processed_data

    def create_binder_labels(self, kd_threshold: float = 1e-7) -> pd.Series:
        """
        Create binary binder labels based on KD values.

        Parameters
        ----------
        kd_threshold : float
            KD threshold for defining binders (default: 100nM)

        Returns
        -------
        pd.Series
            Binary labels (1 for binder, 0 for non-binder)
        """
        if self.processed_data is None:
            self.process_data()

        # Find KD columns
        kd_cols = [col for col in self.processed_data.columns if 'kd_' in col.lower()]

        if not kd_cols:
            print("Warning: No KD columns found. Using binding boolean if available.")
            binding_cols = [col for col in self.processed_data.columns if 'binding_' in col.lower() and 'experimental' in col.lower()]
            if binding_cols:
                return self.processed_data[binding_cols[0]].fillna(0).astype(int)
            return pd.Series([0] * len(self.processed_data))

        # Use minimum KD across all targets
        kd_values = self.processed_data[kd_cols].min(axis=1)
        labels = (kd_values < kd_threshold).astype(int)

        print(f"Created labels: {labels.sum()} binders ({labels.sum()/len(labels)*100:.1f}%), {(~labels.astype(bool)).sum()} non-binders")

        return labels

    def get_numeric_features(self) -> pd.DataFrame:
        """
        Extract only numeric features for modeling.

        Returns
        -------
        pd.DataFrame
            DataFrame with only numeric features
        """
        if self.processed_data is None:
            self.process_data()

        numeric_df = self.processed_data.select_dtypes(include=[np.number])
        print(f"Found {len(numeric_df.columns)} numeric features")

        return numeric_df

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all features.

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if self.processed_data is None:
            self.process_data()

        numeric_df = self.get_numeric_features()

        summary = pd.DataFrame({
            'count': numeric_df.count(),
            'missing': numeric_df.isna().sum(),
            'missing_pct': numeric_df.isna().sum() / len(numeric_df) * 100,
            'mean': numeric_df.mean(),
            'std': numeric_df.std(),
            'min': numeric_df.min(),
            'max': numeric_df.max()
        })

        return summary.sort_values('missing_pct')

    def get_computational_features(self) -> pd.DataFrame:
        """
        Extract only COMPUTATIONAL features (for use as predictors).

        This prevents data leakage by excluding experimental measurements
        when predicting experimental outcomes.

        Returns
        -------
        pd.DataFrame
            DataFrame with only computational features
        """
        if self.raw_data is None:
            self.load_data()

        print("Extracting COMPUTATIONAL features only (preventing data leakage)...")

        # Parse evaluations for each row
        parsed_evaluations = self.raw_data['evaluations'].apply(self.parse_evaluations)

        # Extract ONLY computational metrics
        computational_metrics = parsed_evaluations.apply(
            lambda evals: self.extract_metrics_from_evaluations(evals, filter_type='computational')
        )

        # Convert to DataFrame
        computational_df = pd.DataFrame(computational_metrics.tolist())

        # Add sequence-based features (computational)
        computational_df['sequence_length'] = self.raw_data['sequence'].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )

        # Add basic info columns
        for col in ['id', 'name', 'author', 'designMethod']:
            if col in self.raw_data.columns:
                computational_df[col] = self.raw_data[col]

        # Select only numeric features for modeling
        numeric_cols = computational_df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Extracted {len(numeric_cols)} computational features")
        print(f"Total columns (including metadata): {len(computational_df.columns)}")

        return computational_df

    def get_experimental_features(self) -> pd.DataFrame:
        """
        Extract only EXPERIMENTAL features (for use as labels/targets).

        Returns
        -------
        pd.DataFrame
            DataFrame with only experimental features
        """
        if self.raw_data is None:
            self.load_data()

        print("Extracting EXPERIMENTAL features only...")

        # Parse evaluations for each row
        parsed_evaluations = self.raw_data['evaluations'].apply(self.parse_evaluations)

        # Extract ONLY experimental metrics
        experimental_metrics = parsed_evaluations.apply(
            lambda evals: self.extract_metrics_from_evaluations(evals, filter_type='experimental')
        )

        # Convert to DataFrame
        experimental_df = pd.DataFrame(experimental_metrics.tolist())

        # Add basic info columns
        for col in ['id', 'name']:
            if col in self.raw_data.columns:
                experimental_df[col] = self.raw_data[col]

        print(f"Extracted {len(experimental_df.columns)} experimental feature columns")

        return experimental_df

    def create_binder_labels_from_experimental(self, kd_threshold: float = 1e-7) -> pd.Series:
        """
        Create binary binder labels based on EXPERIMENTAL KD values only.

        This ensures labels come from experimental data, not computational predictions.

        Parameters
        ----------
        kd_threshold : float
            KD threshold for defining binders (default: 100nM = 1e-7 M)

        Returns
        -------
        pd.Series
            Binary labels (1 for binder, 0 for non-binder)
        """
        experimental_df = self.get_experimental_features()

        # Find experimental KD columns
        kd_cols = [col for col in experimental_df.columns
                   if 'kd_' in col.lower() and 'experimental' in col.lower()]

        if not kd_cols:
            print("Warning: No experimental KD columns found. Trying binding boolean...")
            binding_cols = [col for col in experimental_df.columns
                          if 'binding_' in col.lower() and 'experimental' in col.lower()]
            if binding_cols:
                # Use first binding column
                labels = experimental_df[binding_cols[0]].fillna(0).astype(int)
                print(f"Created labels from {binding_cols[0]}")
                print(f"Binders: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%), "
                      f"Non-binders: {(~labels.astype(bool)).sum()} ({(~labels.astype(bool)).sum()/len(labels)*100:.1f}%)")
                return labels

            print("Warning: No experimental binding data found. Returning all zeros.")
            return pd.Series([0] * len(experimental_df))

        # Use minimum KD across all targets (strongest binding)
        kd_values = experimental_df[kd_cols].min(axis=1)
        labels = (kd_values < kd_threshold).astype(int)

        print(f"Created labels from {len(kd_cols)} experimental KD columns")
        print(f"Binders: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%), "
              f"Non-binders: {(~labels.astype(bool)).sum()} ({(~labels.astype(bool)).sum()/len(labels)*100:.1f}%)")

        return labels


def load_and_preprocess(filepath: str, kd_threshold: float = 1e-7) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load and preprocess data in one step.

    DEPRECATED: Use load_computational_and_labels() instead to prevent data leakage.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    kd_threshold : float
        KD threshold for binder classification

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Processed features and labels
    """
    loader = ProteinDataLoader(filepath)
    loader.load_data()
    features = loader.process_data()
    labels = loader.create_binder_labels(kd_threshold)

    return features, labels


def load_computational_and_labels(filepath: str, kd_threshold: float = 1e-7) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load COMPUTATIONAL features and EXPERIMENTAL labels (correct approach).

    This prevents data leakage by:
    - Using only computational features as predictors
    - Using only experimental data as labels/targets

    Parameters
    ----------
    filepath : str
        Path to CSV file
    kd_threshold : float
        KD threshold for binder classification (default: 100nM = 1e-7 M)

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Computational features and experimental labels
    """
    loader = ProteinDataLoader(filepath)
    loader.load_data()

    # Get computational features (predictors)
    computational_features = loader.get_computational_features()

    # Get experimental labels (targets)
    experimental_labels = loader.create_binder_labels_from_experimental(kd_threshold)

    return computational_features, experimental_labels
