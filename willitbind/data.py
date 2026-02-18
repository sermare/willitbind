"""
Data loading and preprocessing for protein binder analysis.

Handles the raw CSV format with JSON-encoded evaluations, separating
computational predictions (features) from experimental measurements (labels)
to prevent data leakage.
"""

import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Amino acid property tables
AA_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "E": 147.13, "Q": 146.15, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
AA_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5,
    "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
CHARGED_POS = set("RKH")
CHARGED_NEG = set("DE")
POLAR = set("STNQYRKHDE")
AROMATIC = set("FWY")
HYDROPHOBIC = set("AILMFVW")


class BinderDataset:
    """Load and prepare the protein binder dataset.

    Parses the raw CSV with JSON-encoded evaluations, splits computational
    features from experimental labels, and computes sequence-derived properties.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw = None
        self._computational = None
        self._experimental = None
        self._analysis_df = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Load the raw CSV and return self for chaining."""
        self.raw = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.raw)} protein designs from {self.filepath}")
        return self

    # ------------------------------------------------------------------
    # Evaluation parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_evals(evals_str):
        try:
            if pd.isna(evals_str) or evals_str == "[]":
                return []
            return json.loads(evals_str)
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _extract_metrics(evals, type_filter=None):
        metrics = {}
        for item in evals:
            etype = item.get("type", "")
            if type_filter and etype != type_filter:
                continue
            metric = item.get("metric", "")
            value = item.get("value")
            target = item.get("target", "")
            suffix = f"_{target}" if target and target != "unknown" else ""
            key = f"{metric}{suffix}_{etype}"
            if isinstance(value, dict):
                if "value" in value:
                    metrics[key] = value["value"]
                elif "metrics" in value:
                    for sub in value["metrics"]:
                        subkey = f"{metric}_{sub.get('slug', '')}{suffix}_{etype}"
                        metrics[subkey] = sub.get("value")
            else:
                metrics[key] = value
        return metrics

    # ------------------------------------------------------------------
    # Computational features (safe to use as predictors)
    # ------------------------------------------------------------------

    def computational_features(self):
        """Return only computational features (no data leakage)."""
        if self._computational is not None:
            return self._computational

        if self.raw is None:
            self.load()

        parsed = self.raw["evaluations"].apply(self._parse_evals)
        comp_metrics = parsed.apply(
            lambda e: self._extract_metrics(e, type_filter="computational")
        )
        comp_df = pd.DataFrame(comp_metrics.tolist(), index=self.raw.index)

        # Add sequence-derived properties
        seqs = self.raw["sequence"].fillna("")
        comp_df["sequence_length"] = seqs.str.len()
        seq_props = seqs.apply(self._sequence_properties)
        prop_df = pd.DataFrame(seq_props.tolist(), index=self.raw.index)
        comp_df = pd.concat([comp_df, prop_df], axis=1)

        # Keep metadata columns for reference
        for col in ["id", "name", "author", "designMethod"]:
            if col in self.raw.columns:
                comp_df[col] = self.raw[col].values

        self._computational = comp_df
        n_numeric = comp_df.select_dtypes(include=[np.number]).shape[1]
        print(f"Extracted {n_numeric} computational features")
        return comp_df

    @staticmethod
    def _sequence_properties(seq):
        """Compute physicochemical properties from amino acid sequence."""
        if not seq or not isinstance(seq, str):
            return {}
        n = len(seq)
        if n == 0:
            return {}
        counts = Counter(seq)
        mw = sum(AA_MW.get(aa, 0) * c for aa, c in counts.items()) - (n - 1) * 18.015
        hydro = np.mean([AA_HYDRO.get(aa, 0) for aa in seq])
        n_pos = sum(counts.get(aa, 0) for aa in CHARGED_POS)
        n_neg = sum(counts.get(aa, 0) for aa in CHARGED_NEG)
        return {
            "mw_from_seq": mw,
            "gravy": hydro,
            "pct_polar": sum(counts.get(aa, 0) for aa in POLAR) / n,
            "pct_hydrophobic": sum(counts.get(aa, 0) for aa in HYDROPHOBIC) / n,
            "pct_aromatic": sum(counts.get(aa, 0) for aa in AROMATIC) / n,
            "pct_charged_pos": n_pos / n,
            "pct_charged_neg": n_neg / n,
            "charge_density": (n_pos - n_neg) / n,
            "net_charge": n_pos - n_neg,
        }

    # ------------------------------------------------------------------
    # Experimental labels (ground truth)
    # ------------------------------------------------------------------

    def experimental_labels(self):
        """Return experimental measurements organized by target."""
        if self._experimental is not None:
            return self._experimental

        if self.raw is None:
            self.load()

        parsed = self.raw["evaluations"].apply(self._parse_evals)
        exp_metrics = parsed.apply(
            lambda e: self._extract_metrics(e, type_filter="experimental")
        )
        exp_df = pd.DataFrame(exp_metrics.tolist(), index=self.raw.index)
        self._experimental = exp_df
        print(f"Extracted {exp_df.shape[1]} experimental measurement columns")
        return exp_df

    # ------------------------------------------------------------------
    # Long-format analysis dataset (protein x target pairs)
    # ------------------------------------------------------------------

    def analysis_dataset(self):
        """Build a long-format dataset: one row per protein-target pair.

        Each row has computational features plus experimental binding
        outcome and affinity for a specific target.
        """
        if self._analysis_df is not None:
            return self._analysis_df

        if self.raw is None:
            self.load()

        comp = self.computational_features()
        comp_numeric = comp.select_dtypes(include=[np.number])

        # Parse experimental binding per target
        parsed = self.raw["evaluations"].apply(self._parse_evals)
        records = []
        for idx, evals in parsed.items():
            targets = {}
            for item in evals:
                if item.get("type") != "experimental":
                    continue
                target = item.get("target")
                if not target or target == "unknown":
                    continue
                if target not in targets:
                    targets[target] = {"binding": None, "kd": None, "pkd": None,
                                       "kon": None, "koff": None}
                metric = item.get("metric")
                value = item.get("value")
                if metric == "binding" and isinstance(value, bool):
                    targets[target]["binding"] = int(value)
                elif metric == "kd" and isinstance(value, (int, float)) and value > 0:
                    targets[target]["kd"] = value
                    targets[target]["pkd"] = -np.log10(value)
                elif metric == "kon" and isinstance(value, (int, float)) and value > 0:
                    targets[target]["kon"] = value
                elif metric == "koff" and isinstance(value, (int, float)) and value > 0:
                    targets[target]["koff"] = value

            for target, bdata in targets.items():
                records.append({
                    "sample_idx": idx,
                    "protein_id": self.raw.at[idx, "id"] if "id" in self.raw.columns else idx,
                    "design_method": self.raw.at[idx, "designMethod"] if "designMethod" in self.raw.columns else "",
                    "target": target,
                    **bdata,
                })

        binding_df = pd.DataFrame(records)
        self._analysis_df = binding_df.merge(
            comp_numeric, left_on="sample_idx", right_index=True, how="left"
        )
        n_binders = (self._analysis_df["binding"] == 1).sum()
        n_total = self._analysis_df["binding"].notna().sum()
        n_targets = self._analysis_df["target"].nunique()
        print(
            f"Analysis dataset: {len(self._analysis_df)} protein-target pairs, "
            f"{n_targets} targets, {n_binders}/{n_total} binders "
            f"({n_binders/n_total*100:.1f}%)"
        )
        return self._analysis_df

    # ------------------------------------------------------------------
    # Convenience: get usable feature names
    # ------------------------------------------------------------------

    def usable_features(self, min_availability=0.10):
        """Return feature names with sufficient data completeness."""
        df = self.analysis_dataset()
        comp = self.computational_features()
        feat_cols = [c for c in comp.select_dtypes(include=[np.number]).columns
                     if c in df.columns]
        avail = pd.Series({c: df[c].notna().mean() for c in feat_cols})
        return avail[avail >= min_availability].sort_values(ascending=False).index.tolist()

    # ------------------------------------------------------------------
    # Binary labels from experimental KD
    # ------------------------------------------------------------------

    def binder_labels(self, kd_threshold=1e-7):
        """Create binary labels: 1 if KD < threshold for any target."""
        exp = self.experimental_labels()
        kd_cols = [c for c in exp.columns if c.startswith("kd_") and "experimental" in c]
        if not kd_cols:
            binding_cols = [c for c in exp.columns if "binding_" in c.lower()]
            if binding_cols:
                return exp[binding_cols[0]].fillna(0).astype(int)
            return pd.Series(0, index=exp.index)
        kd_min = exp[kd_cols].min(axis=1)
        labels = (kd_min < kd_threshold).astype(int)
        print(f"Binary labels: {labels.sum()} binders ({labels.mean()*100:.1f}%) "
              f"at KD < {kd_threshold:.0e}")
        return labels
