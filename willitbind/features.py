"""
Feature analysis for protein binder prediction.

Statistical tests, correlation analysis, feature interaction terms, and
feature selection to identify which computational metrics best predict
experimental binding. Follows the methodology of Overath et al. (2025).
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, kruskal
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class FeatureAnalyzer:
    """Analyze predictive power of computational features."""

    def __init__(self, analysis_df, feature_cols):
        self.df = analysis_df
        self.features = feature_cols

    # ------------------------------------------------------------------
    # Binary binding: binders vs non-binders (Mann-Whitney U)
    # ------------------------------------------------------------------

    def binder_vs_nonbinder(self, min_samples=20, min_per_group=5):
        """Compare feature distributions between binders and non-binders.

        Returns a DataFrame with effect sizes, p-values, and group means,
        sorted by p-value (most significant first).
        """
        bd = self.df[self.df["binding"].notna()].copy()
        results = []
        for feat in self.features:
            sub = bd[[feat, "binding"]].dropna()
            if len(sub) < min_samples:
                continue
            binders = sub.loc[sub["binding"] == 1, feat]
            nonbinders = sub.loc[sub["binding"] == 0, feat]
            if len(binders) < min_per_group or len(nonbinders) < min_per_group:
                continue
            stat, pval = mannwhitneyu(binders, nonbinders, alternative="two-sided")
            pooled_std = np.sqrt((binders.std()**2 + nonbinders.std()**2) / 2)
            d = (binders.mean() - nonbinders.mean()) / pooled_std if pooled_std > 0 else 0.0
            results.append({
                "feature": feat,
                "binder_mean": binders.mean(),
                "binder_std": binders.std(),
                "nonbinder_mean": nonbinders.mean(),
                "nonbinder_std": nonbinders.std(),
                "diff": binders.mean() - nonbinders.mean(),
                "cohens_d": d,
                "p_value": pval,
                "n_binders": len(binders),
                "n_nonbinders": len(nonbinders),
            })
        out = pd.DataFrame(results)
        if len(out):
            out = out.sort_values("p_value").reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Affinity correlation with pKD (Spearman)
    # ------------------------------------------------------------------

    def pkd_correlations(self, min_samples=20):
        """Spearman correlations between features and pKD (binding strength)."""
        pkd = self.df[self.df["pkd"].notna()].copy()
        results = []
        for feat in self.features:
            sub = pkd[[feat, "pkd"]].dropna()
            if len(sub) < min_samples:
                continue
            if sub[feat].std() == 0:
                continue
            rho, pval = spearmanr(sub[feat], sub["pkd"])
            results.append({
                "feature": feat,
                "spearman_r": rho,
                "abs_r": abs(rho),
                "p_value": pval,
                "n_samples": len(sub),
            })
        out = pd.DataFrame(results)
        if len(out):
            out = out.sort_values("abs_r", ascending=False).reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Per-target analysis
    # ------------------------------------------------------------------

    def per_target_stats(self, min_samples=10, top_n_features=5):
        """Run binder vs non-binder and pKD correlation per target."""
        targets = self.df["target"].unique()
        target_info = {}
        for target in targets:
            tdata = self.df[self.df["target"] == target]
            if len(tdata) < min_samples:
                continue
            info = {
                "n_samples": len(tdata),
                "n_binders": int((tdata["binding"] == 1).sum()),
                "n_nonbinders": int((tdata["binding"] == 0).sum()),
                "n_with_kd": int(tdata["kd"].notna().sum()),
            }
            total_labeled = info["n_binders"] + info["n_nonbinders"]
            info["binding_rate"] = info["n_binders"] / total_labeled * 100 if total_labeled else 0

            # Binary stats
            if total_labeled >= min_samples:
                bd = tdata[tdata["binding"].notna()]
                feat_results = []
                for feat in self.features:
                    sub = bd[[feat, "binding"]].dropna()
                    if len(sub) < min_samples:
                        continue
                    b = sub.loc[sub["binding"] == 1, feat]
                    nb = sub.loc[sub["binding"] == 0, feat]
                    if len(b) < 3 or len(nb) < 3:
                        continue
                    _, pval = mannwhitneyu(b, nb, alternative="two-sided")
                    ps = np.sqrt((b.std()**2 + nb.std()**2) / 2)
                    d = (b.mean() - nb.mean()) / ps if ps > 0 else 0.0
                    feat_results.append({"feature": feat, "p_value": pval, "cohens_d": d})
                if feat_results:
                    info["top_binary"] = (
                        pd.DataFrame(feat_results)
                        .sort_values("p_value")
                        .head(top_n_features)
                    )

            # pKD correlations
            if tdata["pkd"].notna().sum() >= min_samples:
                pkd_sub = tdata[tdata["pkd"].notna()]
                info["pkd_mean"] = pkd_sub["pkd"].mean()
                info["pkd_std"] = pkd_sub["pkd"].std()
                corr_results = []
                for feat in self.features:
                    sub = pkd_sub[[feat, "pkd"]].dropna()
                    if len(sub) < 10 or sub[feat].std() == 0:
                        continue
                    rho, pval = spearmanr(sub[feat], sub["pkd"])
                    corr_results.append({"feature": feat, "spearman_r": rho, "p_value": pval})
                if corr_results:
                    info["top_pkd"] = (
                        pd.DataFrame(corr_results)
                        .sort_values("p_value")
                        .head(top_n_features)
                    )

            target_info[target] = info
        return target_info

    # ------------------------------------------------------------------
    # Single feature Average Precision (following Overath et al.)
    # ------------------------------------------------------------------

    def single_feature_ap(self):
        """Rank each feature by its Average Precision for binder classification.

        Following Overath et al., AP is the preferred metric over AUC-ROC
        given the strong class imbalance.
        """
        bd = self.df[self.df["binding"].notna()].copy()
        results = []
        for feat in self.features:
            sub = bd[[feat, "binding"]].dropna()
            if len(sub) < 20 or sub["binding"].nunique() < 2:
                continue
            # For features where higher = more likely binder
            ap_pos = average_precision_score(sub["binding"], sub[feat])
            # For features where lower = more likely binder (flip sign)
            ap_neg = average_precision_score(sub["binding"], -sub[feat])
            if ap_neg > ap_pos:
                ap, direction = ap_neg, "lower_is_better"
            else:
                ap, direction = ap_pos, "higher_is_better"
            results.append({
                "feature": feat,
                "AP": ap,
                "direction": direction,
                "n_samples": len(sub),
                "n_binders": int(sub["binding"].sum()),
            })
        out = pd.DataFrame(results).sort_values("AP", ascending=False).reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Interaction features (pairwise products, as in Overath et al.)
    # ------------------------------------------------------------------

    def interaction_feature_ap(self, top_n_base=15, top_n_interactions=20):
        """Test pairwise feature products for improved AP.

        Overath et al. showed that combining confidence metrics (ipSAE, LIS)
        with physicochemical descriptors (shape complementarity, dG/dSASA)
        through products captures complementary information.
        """
        bd = self.df[self.df["binding"].notna()].copy()

        # Get top base features by AP
        base_ap = self.single_feature_ap()
        top_feats = base_ap.head(top_n_base)["feature"].tolist()

        results = []
        for f1, f2 in combinations(top_feats, 2):
            sub = bd[[f1, f2, "binding"]].dropna()
            if len(sub) < 20:
                continue
            interaction = sub[f1] * sub[f2]
            ap_pos = average_precision_score(sub["binding"], interaction)
            ap_neg = average_precision_score(sub["binding"], -interaction)
            ap = max(ap_pos, ap_neg)
            results.append({
                "feature": f"{f1} x {f2}",
                "f1": f1,
                "f2": f2,
                "AP": ap,
                "n_samples": len(sub),
            })

        out = pd.DataFrame(results).sort_values("AP", ascending=False).reset_index(drop=True)
        print(f"Tested {len(results)} interaction features, best AP: {out['AP'].iloc[0]:.3f}")
        return out.head(top_n_interactions)

    # ------------------------------------------------------------------
    # Per-target AP analysis (as in Overath et al. Fig 3B)
    # ------------------------------------------------------------------

    def per_target_ap(self, feature_name):
        """Compute AP for a given feature per target (as in Overath et al.)."""
        results = []
        for target in self.df["target"].unique():
            tdata = self.df[self.df["target"] == target]
            sub = tdata[[feature_name, "binding"]].dropna()
            if len(sub) < 10 or sub["binding"].nunique() < 2:
                continue
            ap_pos = average_precision_score(sub["binding"], sub[feature_name])
            ap_neg = average_precision_score(sub["binding"], -sub[feature_name])
            ap = max(ap_pos, ap_neg)
            results.append({
                "target": target,
                "AP": ap,
                "n_samples": len(sub),
                "n_binders": int(sub["binding"].sum()),
                "binding_rate": sub["binding"].mean(),
            })
        return pd.DataFrame(results).sort_values("AP", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # LASSO feature selection
    # ------------------------------------------------------------------

    def lasso_select_binary(self, alpha_range=None):
        """LASSO logistic regression for binary binding classification."""
        bd = self.df[self.df["binding"].notna()].copy()
        feats = [f for f in self.features if bd[f].notna().sum() > len(bd) * 0.3]
        X = bd[feats].fillna(bd[feats].median())
        y = bd["binding"].astype(int)
        if y.sum() < 5 or (y == 0).sum() < 5:
            return pd.DataFrame(), pd.DataFrame()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        # Use wider range of C values to avoid over-regularization
        Cs = alpha_range if alpha_range else np.logspace(-4, 2, 30)
        model = LogisticRegressionCV(
            Cs=Cs, penalty="l1", solver="saga", class_weight="balanced",
            cv=5, max_iter=5000, random_state=42,
        )
        model.fit(X_s, y)
        coefs = pd.DataFrame({
            "feature": feats,
            "coefficient": model.coef_[0],
            "abs_coef": np.abs(model.coef_[0]),
        }).sort_values("abs_coef", ascending=False)
        selected = coefs[coefs["abs_coef"] > 0].reset_index(drop=True)
        print(f"LASSO binary: selected {len(selected)}/{len(feats)} features (C={model.C_[0]:.4f})")
        return coefs, selected

    def lasso_select_pkd(self, alpha_range=None):
        """LASSO regression for pKD (binding affinity) prediction."""
        pk = self.df[self.df["pkd"].notna()].copy()
        feats = [f for f in self.features if pk[f].notna().sum() > len(pk) * 0.3]
        X = pk[feats].fillna(pk[feats].median())
        y = pk["pkd"]
        if len(y) < 20:
            return pd.DataFrame()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LassoCV(cv=5, max_iter=5000, random_state=42)
        model.fit(X_s, y)
        coefs = pd.DataFrame({
            "feature": feats,
            "coefficient": model.coef_,
            "abs_coef": np.abs(model.coef_),
        }).sort_values("abs_coef", ascending=False)
        selected = coefs[coefs["abs_coef"] > 0].reset_index(drop=True)
        print(f"LASSO pKD: selected {len(selected)}/{len(feats)} features "
              f"(R2={model.score(X_s, y):.3f})")
        return coefs, selected

    # ------------------------------------------------------------------
    # Feature consistency across targets
    # ------------------------------------------------------------------

    def feature_consistency(self, target_stats):
        """Summarize how often each feature appears in top-N across targets."""
        counts = {}
        for target, info in target_stats.items():
            if "top_binary" in info:
                for feat in info["top_binary"]["feature"].values:
                    counts[feat] = counts.get(feat, 0) + 1
        out = pd.DataFrame([
            {"feature": f, "n_targets_top5": c}
            for f, c in counts.items()
        ]).sort_values("n_targets_top5", ascending=False).reset_index(drop=True)
        return out
