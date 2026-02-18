"""
Feature Engineering Module for Protein Binder Analysis

Implements feature creation including interaction features, confidence metrics,
and physicochemical descriptors as described in the methodology.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Create and transform features for binder prediction."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with base features
        """
        self.data = data.copy()
        self.interaction_features = None
        self.scaler = StandardScaler()

    def create_interaction_features(self, features: List[str], top_k: int = 50) -> pd.DataFrame:
        """
        Create pairwise interaction features (products).

        As described in methodology step 9: Tests pairwise feature products
        to capture non-linear relationships.

        Parameters
        ----------
        features : List[str]
            List of feature names to create interactions from
        top_k : int
            Number of top interaction features to keep (default: 50)

        Returns
        -------
        pd.DataFrame
            DataFrame with interaction features
        """
        print(f"Creating interaction features from {len(features)} base features...")

        # Filter to only numeric features present in data
        valid_features = [f for f in features if f in self.data.columns]
        valid_features = [f for f in valid_features if pd.api.types.is_numeric_dtype(self.data[f])]

        if len(valid_features) < 2:
            print("Warning: Not enough valid numeric features for interactions")
            return pd.DataFrame()

        # Create all pairwise interactions
        interaction_features = {}

        for f1, f2 in combinations(valid_features, 2):
            # Product interaction
            interaction_name = f"{f1}_x_{f2}"
            interaction_features[interaction_name] = self.data[f1] * self.data[f2]

        interaction_df = pd.DataFrame(interaction_features)

        print(f"Created {len(interaction_df.columns)} interaction features")

        # Optionally filter to top_k based on variance or other criteria
        if top_k is not None and top_k < len(interaction_df.columns):
            # Calculate variance for each interaction feature
            variances = interaction_df.var()
            top_features = variances.nlargest(top_k).index
            interaction_df = interaction_df[top_features]
            print(f"Kept top {top_k} interaction features by variance")

        self.interaction_features = interaction_df
        return interaction_df

    def create_confidence_metrics(self) -> pd.DataFrame:
        """
        Create AI-derived confidence score features.

        Simulates metrics from methodology step 5:
        - ipSAE variants (min, max, avg)
        - LIS (Local Interaction Score)
        - pLDDT (per-residue confidence)

        Returns
        -------
        pd.DataFrame
            DataFrame with confidence metric features
        """
        confidence_features = {}

        # Look for pLDDT-like features
        plddt_cols = [col for col in self.data.columns if 'plddt' in col.lower()]

        for col in plddt_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Create variants
                confidence_features[f"{col}_normalized"] = self.data[col] / 100.0

        # Look for confidence scores
        confidence_cols = [col for col in self.data.columns if 'confidence' in col.lower()]
        for col in confidence_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                confidence_features[f"{col}_squared"] = self.data[col] ** 2

        return pd.DataFrame(confidence_features)

    def create_physicochemical_features(self) -> pd.DataFrame:
        """
        Create physicochemical descriptor features.

        Based on methodology step 7: interface energetic and geometric properties.

        Returns
        -------
        pd.DataFrame
            DataFrame with physicochemical features
        """
        physico_features = {}

        # Sequence length features
        if 'sequence_length' in self.data.columns:
            physico_features['log_sequence_length'] = np.log1p(self.data['sequence_length'])
            physico_features['sequence_length_squared'] = self.data['sequence_length'] ** 2

        # Molecular weight features
        mw_cols = [col for col in self.data.columns if 'molecular_weight' in col.lower()]
        for col in mw_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                physico_features['log_' + col] = np.log1p(self.data[col])

        # Isoelectric point features
        ip_cols = [col for col in self.data.columns if 'isoelectric_point' in col.lower()]
        for col in ip_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Deviation from neutral pH
                physico_features[f"{col}_deviation_from_7"] = np.abs(self.data[col] - 7.0)

        return pd.DataFrame(physico_features)

    def create_kinetic_features(self) -> pd.DataFrame:
        """
        Create features from kinetic binding data (Kon, Koff, KD).

        Returns
        -------
        pd.DataFrame
            DataFrame with kinetic features
        """
        kinetic_features = {}

        # Find kinetic columns
        kon_cols = [col for col in self.data.columns if 'kon_' in col.lower()]
        koff_cols = [col for col in self.data.columns if 'koff_' in col.lower()]
        kd_cols = [col for col in self.data.columns if 'kd_' in col.lower()]

        # Log-transform kinetic parameters
        for col in kon_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                kinetic_features[f"log_{col}"] = np.log10(self.data[col].replace(0, np.nan))

        for col in koff_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                kinetic_features[f"log_{col}"] = np.log10(self.data[col].replace(0, np.nan))

        for col in kd_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                kinetic_features[f"log_{col}"] = np.log10(self.data[col].replace(0, np.nan))
                kinetic_features[f"pKd_{col}"] = -np.log10(self.data[col].replace(0, np.nan))

        return pd.DataFrame(kinetic_features)

    def create_all_engineered_features(self, include_interactions: bool = True,
                                        top_k_interactions: int = 50) -> pd.DataFrame:
        """
        Create all engineered features.

        Parameters
        ----------
        include_interactions : bool
            Whether to include interaction features
        top_k_interactions : int
            Number of top interaction features to keep

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all engineered features
        """
        print("Creating all engineered features...")

        feature_dfs = []

        # Confidence metrics
        conf_features = self.create_confidence_metrics()
        if not conf_features.empty:
            feature_dfs.append(conf_features)
            print(f"  - {len(conf_features.columns)} confidence features")

        # Physicochemical features
        physico_features = self.create_physicochemical_features()
        if not physico_features.empty:
            feature_dfs.append(physico_features)
            print(f"  - {len(physico_features.columns)} physicochemical features")

        # Kinetic features
        kinetic_features = self.create_kinetic_features()
        if not kinetic_features.empty:
            feature_dfs.append(kinetic_features)
            print(f"  - {len(kinetic_features.columns)} kinetic features")

        # Interaction features
        if include_interactions:
            # Get numeric features for interactions
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            interaction_features = self.create_interaction_features(
                numeric_cols, top_k=top_k_interactions
            )
            if not interaction_features.empty:
                feature_dfs.append(interaction_features)
                print(f"  - {len(interaction_features.columns)} interaction features")

        # Combine all features
        if feature_dfs:
            all_features = pd.concat([self.data] + feature_dfs, axis=1)
        else:
            all_features = self.data

        print(f"Total features after engineering: {len(all_features.columns)}")

        return all_features

    def select_features_by_correlation(self, target: pd.Series,
                                        threshold: float = 0.01) -> List[str]:
        """
        Select features based on correlation with target.

        Parameters
        ----------
        target : pd.Series
            Target variable
        threshold : float
            Minimum absolute correlation to keep feature

        Returns
        -------
        List[str]
            List of selected feature names
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        # Calculate correlations
        correlations = numeric_data.corrwith(target).abs()

        # Select features above threshold
        selected = correlations[correlations > threshold].index.tolist()

        print(f"Selected {len(selected)} features with |correlation| > {threshold}")

        return selected

    def normalize_features(self, features: pd.DataFrame,
                           fit: bool = True) -> pd.DataFrame:
        """
        Z-score normalize features.

        Parameters
        ----------
        features : pd.DataFrame
            Features to normalize
        fit : bool
            Whether to fit the scaler (True) or use existing fit (False)

        Returns
        -------
        pd.DataFrame
            Normalized features
        """
        if fit:
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)

        return pd.DataFrame(normalized, columns=features.columns, index=features.index)


def create_simulated_af_features(n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Create simulated AlphaFold-like confidence features for demonstration.

    This simulates the features from AF2, AF3, ColabFold, and Boltz-1
    as described in the methodology.

    Parameters
    ----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated features
    """
    np.random.seed(random_state)

    features = {}

    # Simulate features for each model
    models = ['AF2', 'AF3', 'ColabFold', 'Boltz1']

    for model in models:
        # ipSAE variants (0-1, higher is better)
        features[f'{model}_ipSAE_min'] = np.random.beta(2, 2, n_samples)
        features[f'{model}_ipSAE_max'] = np.random.beta(2, 2, n_samples)
        features[f'{model}_ipSAE_avg'] = (features[f'{model}_ipSAE_min'] + features[f'{model}_ipSAE_max']) / 2

        # LIS (Local Interaction Score)
        features[f'{model}_LIS'] = np.random.beta(2, 2, n_samples)

        # ipAE (0-30, lower is better)
        features[f'{model}_ipAE'] = np.random.gamma(2, 5, n_samples)

        # ipTM (0-1, higher is better)
        features[f'{model}_ipTM'] = np.random.beta(5, 2, n_samples)

        # pLDDT (0-100, higher is better)
        features[f'{model}_pLDDT'] = np.random.beta(5, 2, n_samples) * 100

        # pDockQ (0-1, higher is better)
        features[f'{model}_pDockQ'] = np.random.beta(3, 2, n_samples)

        # RMSD (0-10, lower is better)
        features[f'RMSD_binder_{model}'] = np.random.gamma(2, 1, n_samples)
        features[f'RMSD_complex_{model}'] = np.random.gamma(2, 1.5, n_samples)

        # DockQ (0-1, higher is better)
        features[f'DockQ_{model}'] = np.random.beta(3, 2, n_samples)

    # Input structure features (Rosetta)
    features['input_interface_dG'] = np.random.normal(-15, 5, n_samples)
    features['input_interface_dG_dSASA'] = features['input_interface_dG'] / np.random.uniform(500, 2000, n_samples)
    features['input_interface_shape_complementarity'] = np.random.beta(5, 2, n_samples)
    features['input_packstat'] = np.random.beta(5, 2, n_samples)
    features['input_dSASA'] = np.random.uniform(500, 2000, n_samples)
    features['input_interface_hbonds'] = np.random.poisson(5, n_samples)
    features['input_deltaSAP'] = np.random.normal(0, 5, n_samples)

    return pd.DataFrame(features)
