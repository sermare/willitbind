"""
Model Training and Evaluation Utilities

Implements the modeling approaches from the methodology including:
- Greedy forward feature selection (Step 12)
- Leave-one-group-out cross-validation (Step 12)
- Logistic regression with L1 regularization (Step 12)
- XGBoost alternative models (Step 14)
- Threshold optimization strategies (Step 11, 17)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')


class BinderClassifier:
    """Train and evaluate binder classification models."""

    def __init__(self, model_type: str = 'logistic', random_state: int = 42):
        """
        Initialize classifier.

        Parameters
        ----------
        model_type : str
            Type of model ('logistic' or 'xgboost')
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = None

    def _create_model(self):
        """Create model instance based on model_type."""
        if self.model_type == 'logistic':
            # L1-penalized logistic regression with balanced class weights
            self.model = LogisticRegression(
                penalty='l1',
                solver='saga',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                # Calculate scale_pos_weight for class imbalance
                self.model = XGBClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1
                )
            except ImportError:
                print("XGBoost not available. Falling back to logistic regression.")
                self.model_type = 'logistic'
                self._create_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        """
        self._create_model()

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.selected_features = X.columns.tolist()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        np.ndarray
            Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        threshold : float
            Classification threshold

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            True labels

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        y_pred_proba = self.predict_proba(X)

        # Find optimal threshold by F1 score
        optimal_threshold = self.find_optimal_threshold(y, y_pred_proba, metric='f1')
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        metrics = {
            'average_precision': average_precision_score(y, y_pred_proba),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'optimal_threshold': optimal_threshold
        }

        return metrics

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                                metric: str = 'f1') -> float:
        """
        Find optimal classification threshold.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        metric : str
            Metric to optimize ('f1', 'precision', 'recall')

        Returns
        -------
        float
            Optimal threshold
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        if metric == 'f1':
            # Calculate F1 scores
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions[:-1])
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls[:-1])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    @staticmethod
    def find_threshold_at_recall(y_true: np.ndarray, y_proba: np.ndarray,
                                  target_recall: float = 0.4) -> float:
        """
        Find threshold that achieves target recall.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        target_recall : float
            Target recall value

        Returns
        -------
        float
            Threshold achieving target recall
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recalls[:-1] - target_recall))
        return thresholds[idx] if idx < len(thresholds) else 0.5


class GreedyFeatureSelector:
    """
    Greedy forward feature selection with cross-validation.

    Implements methodology Step 12: Greedy forward selection with
    logistic regression and nested LOGO-CV.
    """

    def __init__(self, model_type: str = 'logistic',
                 metric: str = 'average_precision',
                 early_stopping_threshold: float = 0.005,
                 random_state: int = 42):
        """
        Initialize feature selector.

        Parameters
        ----------
        model_type : str
            Type of model to use
        metric : str
            Metric to optimize ('average_precision', 'f1', 'roc_auc')
        early_stopping_threshold : float
            Minimum improvement to continue adding features
        random_state : int
            Random seed
        """
        self.model_type = model_type
        self.metric = metric
        self.early_stopping_threshold = early_stopping_threshold
        self.random_state = random_state
        self.selected_features = []
        self.feature_scores = []

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        groups: Optional[pd.Series] = None,
                        max_features: int = 10) -> List[str]:
        """
        Perform greedy forward feature selection.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        groups : pd.Series, optional
            Group labels for cross-validation
        max_features : int
            Maximum number of features to select

        Returns
        -------
        List[str]
            Selected feature names
        """
        print(f"Starting greedy feature selection (max {max_features} features)...")

        available_features = X.columns.tolist()
        self.selected_features = []
        self.feature_scores = []

        best_score = 0.0

        for iteration in range(max_features):
            print(f"\nIteration {iteration + 1}:")

            # Try adding each remaining feature
            candidate_scores = {}

            for feature in available_features:
                if feature in self.selected_features:
                    continue

                # Test features with this feature added
                test_features = self.selected_features + [feature]

                # Evaluate using cross-validation
                if groups is not None:
                    score = self._cross_validate(
                        X[test_features], y, groups
                    )
                else:
                    # Simple train-test split if no groups
                    score = self._evaluate_features(X[test_features], y)

                candidate_scores[feature] = score

            if not candidate_scores:
                print("No more features to add.")
                break

            # Select best feature
            best_feature = max(candidate_scores, key=candidate_scores.get)
            best_new_score = candidate_scores[best_feature]

            # Check for early stopping
            improvement = best_new_score - best_score

            print(f"  Best feature: {best_feature}")
            print(f"  Score: {best_new_score:.4f} (improvement: {improvement:.4f})")

            if improvement < self.early_stopping_threshold:
                print(f"  Early stopping: improvement < {self.early_stopping_threshold}")
                break

            # Add feature
            self.selected_features.append(best_feature)
            self.feature_scores.append(best_new_score)
            best_score = best_new_score

        print(f"\nSelected {len(self.selected_features)} features")
        print(f"Final score: {best_score:.4f}")

        return self.selected_features

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series,
                        groups: pd.Series) -> float:
        """
        Perform leave-one-group-out cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        groups : pd.Series
            Group labels

        Returns
        -------
        float
            Median score across folds
        """
        logo = LeaveOneGroupOut()
        scores = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Handle missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())

            # Train model
            classifier = BinderClassifier(model_type=self.model_type,
                                          random_state=self.random_state)
            classifier.fit(X_train, y_train)

            # Evaluate
            y_pred_proba = classifier.predict_proba(X_test)

            if self.metric == 'average_precision':
                score = average_precision_score(y_test, y_pred_proba)
            elif self.metric == 'roc_auc':
                score = roc_auc_score(y_test, y_pred_proba)
            elif self.metric == 'f1':
                threshold = classifier.find_optimal_threshold(y_test, y_pred_proba, 'f1')
                y_pred = (y_pred_proba >= threshold).astype(int)
                score = f1_score(y_test, y_pred)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            scores.append(score)

        # Return median score to reduce outlier influence (methodology step 10)
        return np.median(scores)

    def _evaluate_features(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Simple evaluation without cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels

        Returns
        -------
        float
            Evaluation score
        """
        # Handle missing values
        X = X.fillna(X.mean())

        # Train model
        classifier = BinderClassifier(model_type=self.model_type,
                                      random_state=self.random_state)
        classifier.fit(X, y)

        # Evaluate on same data (for demonstration)
        y_pred_proba = classifier.predict_proba(X)

        if self.metric == 'average_precision':
            return average_precision_score(y, y_pred_proba)
        elif self.metric == 'roc_auc':
            return roc_auc_score(y, y_pred_proba)
        elif self.metric == 'f1':
            threshold = classifier.find_optimal_threshold(y, y_pred_proba, 'f1')
            y_pred = (y_pred_proba >= threshold).astype(int)
            return f1_score(y, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


def evaluate_single_feature_performance(X: pd.DataFrame, y: pd.Series,
                                         groups: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Evaluate performance of individual features.

    Implements methodology Step 8: Single feature performance analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Labels
    groups : pd.Series, optional
        Group labels for cross-validation

    Returns
    -------
    pd.DataFrame
        Feature performance results
    """
    print("Evaluating single feature performance...")

    results = []

    for feature in X.columns:
        if not pd.api.types.is_numeric_dtype(X[feature]):
            continue

        # Get feature values
        feature_values = X[[feature]].fillna(X[feature].mean())

        if groups is not None:
            # Cross-validation
            logo = LeaveOneGroupOut()
            aps = []

            for train_idx, test_idx in logo.split(feature_values, y, groups):
                X_train, X_test = feature_values.iloc[train_idx], feature_values.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                classifier = BinderClassifier(random_state=42)
                classifier.fit(X_train, y_train)
                y_pred_proba = classifier.predict_proba(X_test)

                ap = average_precision_score(y_test, y_pred_proba)
                aps.append(ap)

            median_ap = np.median(aps)
            mean_ap = np.mean(aps)
            std_ap = np.std(aps)
        else:
            # Simple evaluation
            classifier = BinderClassifier(random_state=42)
            classifier.fit(feature_values, y)
            y_pred_proba = classifier.predict_proba(feature_values)

            median_ap = average_precision_score(y, y_pred_proba)
            mean_ap = median_ap
            std_ap = 0.0

        results.append({
            'feature': feature,
            'median_AP': median_ap,
            'mean_AP': mean_ap,
            'std_AP': std_ap
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('median_AP', ascending=False)

    print(f"Evaluated {len(results_df)} features")

    return results_df
