"""
Predictive modeling for protein binder classification.

Logistic regression and greedy feature selection with proper cross-validation
to build models that predict binding from computational features alone.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class BindingPredictor:
    """Train and evaluate a binder classification model."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.features_used = None

    def fit(self, X, y):
        self.model = LogisticRegression(
            penalty="l1", solver="saga", class_weight="balanced",
            random_state=self.random_state, max_iter=2000, C=1.0,
        )
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)
        self.features_used = list(X.columns)

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        thr = optimal_threshold(y, proba)
        pred = (proba >= thr).astype(int)
        return {
            "average_precision": average_precision_score(y, proba),
            "roc_auc": roc_auc_score(y, proba),
            "f1": f1_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "threshold": thr,
        }

    def coefficients(self):
        if self.model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self.features_used,
            "coefficient": self.model.coef_[0],
            "abs_coef": np.abs(self.model.coef_[0]),
        }).sort_values("abs_coef", ascending=False)


class GreedySelector:
    """Greedy forward feature selection with cross-validation."""

    def __init__(self, metric="average_precision", early_stop=0.005, random_state=42):
        self.metric = metric
        self.early_stop = early_stop
        self.random_state = random_state
        self.selected = []
        self.scores = []

    def select(self, X, y, groups=None, max_features=10):
        self.selected = []
        self.scores = []
        best_score = 0.0
        candidates = list(X.columns)

        for step in range(max_features):
            step_scores = {}
            for feat in candidates:
                if feat in self.selected:
                    continue
                trial = self.selected + [feat]
                score = self._cv_score(X[trial], y, groups)
                step_scores[feat] = score

            if not step_scores:
                break

            best_feat = max(step_scores, key=step_scores.get)
            improvement = step_scores[best_feat] - best_score

            if improvement < self.early_stop and step > 0:
                print(f"  Step {step+1}: early stop (improvement {improvement:.4f} < {self.early_stop})")
                break

            self.selected.append(best_feat)
            self.scores.append(step_scores[best_feat])
            best_score = step_scores[best_feat]
            print(f"  Step {step+1}: +{best_feat} -> {best_score:.4f} (+{improvement:.4f})")

        return self.selected

    def _cv_score(self, X, y, groups):
        X_filled = X.fillna(X.mean())
        if groups is not None and groups.nunique() > 1:
            logo = LeaveOneGroupOut()
            fold_scores = []
            for tr, te in logo.split(X_filled, y, groups):
                Xtr, Xte = X_filled.iloc[tr], X_filled.iloc[te]
                ytr, yte = y.iloc[tr], y.iloc[te]
                if yte.nunique() < 2:
                    continue
                m = BindingPredictor(self.random_state)
                m.fit(Xtr, ytr)
                proba = m.predict_proba(Xte)
                fold_scores.append(average_precision_score(yte, proba))
            return np.median(fold_scores) if fold_scores else 0.0
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            fold_scores = []
            for tr, te in skf.split(X_filled, y):
                Xtr, Xte = X_filled.iloc[tr], X_filled.iloc[te]
                ytr, yte = y.iloc[tr], y.iloc[te]
                m = BindingPredictor(self.random_state)
                m.fit(Xtr, ytr)
                proba = m.predict_proba(Xte)
                fold_scores.append(average_precision_score(yte, proba))
            return np.median(fold_scores) if fold_scores else 0.0


def optimal_threshold(y_true, y_proba, metric="f1"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    idx = np.argmax(f1[:-1])
    return thresholds[idx] if idx < len(thresholds) else 0.5


def single_feature_performance(X, y, groups=None, random_state=42):
    """Evaluate each feature individually and return a ranked DataFrame."""
    results = []
    for feat in X.columns:
        if not pd.api.types.is_numeric_dtype(X[feat]):
            continue
        Xf = X[[feat]].fillna(X[feat].mean())
        if groups is not None and groups.nunique() > 1:
            logo = LeaveOneGroupOut()
            aps = []
            for tr, te in logo.split(Xf, y, groups):
                ytr, yte = y.iloc[tr], y.iloc[te]
                if yte.nunique() < 2:
                    continue
                m = BindingPredictor(random_state)
                m.fit(Xf.iloc[tr], ytr)
                proba = m.predict_proba(Xf.iloc[te])
                aps.append(average_precision_score(yte, proba))
            results.append({
                "feature": feat,
                "median_AP": np.median(aps) if aps else 0,
                "mean_AP": np.mean(aps) if aps else 0,
                "std_AP": np.std(aps) if aps else 0,
            })
        else:
            m = BindingPredictor(random_state)
            m.fit(Xf, y)
            proba = m.predict_proba(Xf)
            ap = average_precision_score(y, proba)
            results.append({"feature": feat, "median_AP": ap, "mean_AP": ap, "std_AP": 0.0})
    return pd.DataFrame(results).sort_values("median_AP", ascending=False).reset_index(drop=True)
