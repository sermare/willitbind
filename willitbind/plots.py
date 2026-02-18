"""
Publication-quality visualizations for protein binder analysis.

All plots use a consistent style with no bold text annotations and no
em-dashes. Designed to be informative and clean for both papers and READMEs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

# Consistent style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
})

PALETTE = {
    "binder": "#2ecc71",
    "nonbinder": "#e74c3c",
    "primary": "#3498db",
    "secondary": "#e67e22",
    "accent": "#9b59b6",
    "dark": "#2c3e50",
    "light": "#ecf0f1",
}


class WillItPlot:
    """Generate all analysis figures."""

    def __init__(self, output_dir="results/figures"):
        self.output_dir = output_dir

    def _savefig(self, fig, name):
        path = f"{self.output_dir}/{name}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Figure 1: Dataset overview
    # ------------------------------------------------------------------

    def dataset_overview(self, analysis_df, usable_features):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

        # A: Samples per target
        ax = fig.add_subplot(gs[0, :2])
        tc = analysis_df["target"].value_counts().head(15)
        colors = sns.color_palette("husl", len(tc))
        ax.barh(range(len(tc)), tc.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(tc)))
        ax.set_yticklabels([t[:35] for t in tc.index], fontsize=9)
        ax.set_xlabel("Number of protein-target pairs")
        ax.set_title("A. Samples per target")
        ax.invert_yaxis()

        # B: Binding rate per target
        ax = fig.add_subplot(gs[0, 2])
        bd = analysis_df[analysis_df["binding"].notna()]
        rates = bd.groupby("target")["binding"].mean().sort_values(ascending=False).head(15) * 100
        cmap = ["#2ecc71" if r > 50 else "#e67e22" if r > 25 else "#e74c3c" for r in rates]
        ax.barh(range(len(rates)), rates.values, color=cmap, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([t[:25] for t in rates.index], fontsize=8)
        ax.set_xlabel("Binding rate (%)")
        ax.set_title("B. Success rate by target")
        ax.axvline(50, color="gray", ls="--", alpha=0.5, lw=1)
        ax.invert_yaxis()

        # C: pKD distribution
        ax = fig.add_subplot(gs[1, 0])
        pkd = analysis_df["pkd"].dropna()
        ax.hist(pkd, bins=30, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
        ax.axvline(pkd.median(), color=PALETTE["nonbinder"], ls="--", lw=1.5,
                   label=f"Median: {pkd.median():.1f}")
        ax.set_xlabel("pKD (-log10 KD)")
        ax.set_ylabel("Count")
        ax.set_title("C. Affinity distribution")
        ax.legend(frameon=True, fontsize=9)

        # D: KD log-scale
        ax = fig.add_subplot(gs[1, 1])
        kd = analysis_df["kd"].dropna()
        ax.hist(np.log10(kd), bins=30, color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
        ax.axvline(np.log10(1e-9), color="#2ecc71", ls="--", lw=1.5, label="1 nM")
        ax.axvline(np.log10(1e-7), color="#e67e22", ls="--", lw=1.5, label="100 nM")
        ax.set_xlabel("log10(KD) [M]")
        ax.set_ylabel("Count")
        ax.set_title("D. KD distribution")
        ax.legend(frameon=True, fontsize=9)

        # E: Class balance pie
        ax = fig.add_subplot(gs[1, 2])
        bc = bd["binding"].value_counts().sort_index()
        labels = ["Non-binder", "Binder"]
        colors_pie = [PALETTE["nonbinder"], PALETTE["binder"]]
        wedges, texts, autotexts = ax.pie(
            bc.values, labels=labels, autopct="%1.1f%%", colors=colors_pie,
            startangle=90, textprops={"fontsize": 10},
        )
        ax.set_title("E. Class balance")

        # F: Feature completeness
        ax = fig.add_subplot(gs[2, :])
        top_feats = usable_features[:20]
        completeness = [analysis_df[f].notna().mean() * 100 for f in top_feats]
        cvals = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_feats)))
        ax.barh(range(len(top_feats)), completeness, color=cvals, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels([f[:50] for f in top_feats], fontsize=8)
        ax.set_xlabel("Data completeness (%)")
        ax.set_title("F. Feature availability (top 20)")
        ax.axvline(50, color="red", ls="--", alpha=0.4, lw=1)
        ax.invert_yaxis()

        fig.suptitle("Dataset Overview: 5,000+ Protein Binder Designs", fontsize=14, y=1.01)
        return self._savefig(fig, "fig1_dataset_overview.png")

    # ------------------------------------------------------------------
    # Figure 2: Effect sizes (binder vs non-binder)
    # ------------------------------------------------------------------

    def effect_sizes(self, stat_df, top_n=20):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]})

        df = stat_df.head(top_n).copy()
        df = df.sort_values("cohens_d")

        # Left: effect sizes
        ax = axes[0]
        colors = [PALETTE["binder"] if d > 0 else PALETTE["nonbinder"] for d in df["cohens_d"]]
        y = np.arange(len(df))
        ax.barh(y, df["cohens_d"], color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f[:45] for f in df["feature"]], fontsize=9)
        ax.set_xlabel("Cohen's d (effect size)")
        ax.set_title("A. Effect sizes: binders vs non-binders")
        ax.axvline(0, color="black", lw=1)
        ax.axvline(0.5, color="orange", ls="--", alpha=0.4, label="|d| = 0.5 (medium)")
        ax.axvline(-0.5, color="orange", ls="--", alpha=0.4)
        ax.axvline(0.8, color="red", ls="--", alpha=0.4, label="|d| = 0.8 (large)")
        ax.axvline(-0.8, color="red", ls="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8)

        # Right: p-values
        ax = axes[1]
        logp = -np.log10(df["p_value"].clip(lower=1e-300))
        ax.barh(y, logp, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(df))),
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([""] * len(df))
        ax.set_xlabel("-log10(p-value)")
        ax.set_title("B. Statistical significance")
        ax.axvline(-np.log10(0.05), color="orange", ls="--", lw=1.5, label="p = 0.05")
        ax.axvline(-np.log10(0.001), color="red", ls="--", lw=1.5, label="p = 0.001")
        ax.legend(fontsize=8)

        fig.suptitle("Statistical Comparison: Which Features Separate Binders?", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig2_effect_sizes.png")

    # ------------------------------------------------------------------
    # Figure 3: Volcano plot
    # ------------------------------------------------------------------

    def volcano(self, stat_df):
        fig, ax = plt.subplots(figsize=(10, 8))
        df = stat_df.copy()
        logp = -np.log10(df["p_value"].clip(lower=1e-300))

        sig_large = (df["p_value"] < 0.05) & (df["cohens_d"].abs() > 0.5)
        sig_small = (df["p_value"] < 0.05) & (df["cohens_d"].abs() <= 0.5)
        ns = ~(df["p_value"] < 0.05)

        ax.scatter(df.loc[ns, "cohens_d"], logp[ns], c="#bdc3c7", s=40, alpha=0.5,
                   edgecolor="white", linewidth=0.3, label="Not significant")
        ax.scatter(df.loc[sig_small, "cohens_d"], logp[sig_small], c=PALETTE["secondary"],
                   s=50, alpha=0.7, edgecolor="white", linewidth=0.3, label="Significant, small effect")
        ax.scatter(df.loc[sig_large, "cohens_d"], logp[sig_large], c=PALETTE["nonbinder"],
                   s=70, alpha=0.8, edgecolor="black", linewidth=0.5, label="Significant, large effect")

        # Label top hits
        top = df.nsmallest(5, "p_value")
        for _, row in top.iterrows():
            lp = -np.log10(max(row["p_value"], 1e-300))
            ax.annotate(row["feature"][:30], (row["cohens_d"], lp),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        ax.axhline(-np.log10(0.05), color="gray", ls="--", alpha=0.5, lw=1)
        ax.axvline(0.5, color="gray", ls="--", alpha=0.3)
        ax.axvline(-0.5, color="gray", ls="--", alpha=0.3)
        ax.set_xlabel("Cohen's d (effect size)")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("Volcano Plot: Significance vs Effect Size")
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        return self._savefig(fig, "fig3_volcano.png")

    # ------------------------------------------------------------------
    # Figure 4: pKD correlations
    # ------------------------------------------------------------------

    def pkd_correlation_bars(self, corr_df, top_n=20):
        fig, ax = plt.subplots(figsize=(12, 8))
        df = corr_df.head(top_n).copy().sort_values("spearman_r")
        y = np.arange(len(df))
        colors = [PALETTE["binder"] if r > 0 else PALETTE["nonbinder"] for r in df["spearman_r"]]
        ax.barh(y, df["spearman_r"], color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f[:45] for f in df["feature"]], fontsize=9)
        ax.set_xlabel("Spearman correlation with pKD")
        ax.set_title("Features Correlated with Binding Strength")
        ax.axvline(0, color="black", lw=1)
        ax.axvline(0.3, color="orange", ls="--", alpha=0.4, label="|r| = 0.3")
        ax.axvline(-0.3, color="orange", ls="--", alpha=0.4)
        ax.axvline(0.5, color="red", ls="--", alpha=0.4, label="|r| = 0.5")
        ax.axvline(-0.5, color="red", ls="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)
        return self._savefig(fig, "fig4_pkd_correlations.png")

    # ------------------------------------------------------------------
    # Figure 5: Top feature violin plots
    # ------------------------------------------------------------------

    def top_feature_violins(self, analysis_df, stat_df, n_features=6):
        top = stat_df.head(n_features)
        ncols = 3
        nrows = (n_features + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten() if n_features > 1 else [axes]

        bd = analysis_df[analysis_df["binding"].notna()].copy()
        bd["label"] = bd["binding"].map({0: "Non-binder", 1: "Binder"})

        for i, (_, row) in enumerate(top.iterrows()):
            ax = axes[i]
            feat = row["feature"]
            sub = bd[[feat, "label"]].dropna()
            parts = ax.violinplot(
                [sub.loc[sub["label"] == "Non-binder", feat].values,
                 sub.loc[sub["label"] == "Binder", feat].values],
                positions=[0, 1], showmeans=True, showmedians=True,
            )
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(PALETTE["nonbinder"] if j == 0 else PALETTE["binder"])
                pc.set_alpha(0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-binder", "Binder"])
            ax.set_title(feat[:40], fontsize=10)
            ax.set_ylabel("Value")
            # Annotate with stats
            ax.text(0.5, 0.97, f"d = {row['cohens_d']:.2f}, p = {row['p_value']:.1e}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6))

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Top Discriminative Features: Binder vs Non-binder Distributions", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig5_top_violins.png")

    # ------------------------------------------------------------------
    # Figure 6: LASSO coefficients
    # ------------------------------------------------------------------

    def lasso_coefficients(self, all_coefs, selected, task="binary"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.5]})

        # Left: selected features
        ax = axes[0]
        if len(selected) > 0:
            sel = selected.sort_values("coefficient")
            colors = [PALETTE["binder"] if c > 0 else PALETTE["nonbinder"] for c in sel["coefficient"]]
            y = np.arange(len(sel))
            ax.barh(y, sel["coefficient"], color=colors, alpha=0.8, edgecolor="white")
            ax.set_yticks(y)
            ax.set_yticklabels([f[:40] for f in sel["feature"]], fontsize=9)
        ax.set_xlabel("LASSO coefficient")
        label = "binding" if task == "binary" else "pKD"
        ax.set_title(f"A. Selected features ({label})")
        ax.axvline(0, color="black", lw=1)

        # Right: coefficient path
        ax = axes[1]
        nonzero = all_coefs[all_coefs["abs_coef"] > 0].sort_values("abs_coef", ascending=False).head(15)
        if len(nonzero) > 0:
            colors = [PALETTE["binder"] if c > 0 else PALETTE["nonbinder"] for c in nonzero["coefficient"]]
            y = np.arange(len(nonzero))
            ax.barh(y, nonzero["coefficient"], color=colors, alpha=0.8, edgecolor="white")
            ax.set_yticks(y)
            ax.set_yticklabels([f[:40] for f in nonzero["feature"]], fontsize=9)
        ax.set_xlabel("LASSO coefficient")
        ax.set_title(f"B. All non-zero coefficients ({label})")
        ax.axvline(0, color="black", lw=1)

        fig.suptitle(f"LASSO Feature Selection for {label.title()} Prediction", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, f"fig6_lasso_{task}.png")

    # ------------------------------------------------------------------
    # Figure 7: Per-target comparison
    # ------------------------------------------------------------------

    def per_target_comparison(self, target_stats):
        targets = []
        for t, info in sorted(target_stats.items(), key=lambda x: x[1]["n_samples"], reverse=True):
            if info["n_binders"] + info["n_nonbinders"] >= 10:
                targets.append({
                    "target": t,
                    "n_samples": info["n_samples"],
                    "binding_rate": info["binding_rate"],
                    "n_binders": info["n_binders"],
                    "top_feature": info.get("top_binary", pd.DataFrame()).iloc[0]["feature"][:30]
                    if "top_binary" in info and len(info["top_binary"]) > 0 else "N/A",
                    "top_effect": info.get("top_binary", pd.DataFrame()).iloc[0]["cohens_d"]
                    if "top_binary" in info and len(info["top_binary"]) > 0 else 0,
                })

        if not targets:
            return None

        df = pd.DataFrame(targets)
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        # A: Sample sizes
        ax = axes[0]
        colors = sns.color_palette("husl", len(df))
        ax.barh(range(len(df)), df["n_samples"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([t[:30] for t in df["target"]], fontsize=9)
        ax.set_xlabel("Sample count")
        ax.set_title("A. Samples per target")
        ax.invert_yaxis()

        # B: Binding rates
        ax = axes[1]
        cmap = ["#2ecc71" if r > 50 else "#e67e22" if r > 25 else "#e74c3c" for r in df["binding_rate"]]
        ax.barh(range(len(df)), df["binding_rate"], color=cmap, edgecolor="white")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([""] * len(df))
        ax.set_xlabel("Binding rate (%)")
        ax.set_title("B. Binding success rate")
        ax.axvline(50, color="gray", ls="--", alpha=0.5)
        ax.invert_yaxis()

        # C: Top effect sizes
        ax = axes[2]
        cmap2 = [PALETTE["binder"] if e > 0 else PALETTE["nonbinder"] for e in df["top_effect"]]
        ax.barh(range(len(df)), df["top_effect"], color=cmap2, edgecolor="white")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([""] * len(df))
        ax.set_xlabel("Best feature effect size (d)")
        ax.set_title("C. Best computational predictor per target")
        ax.axvline(0, color="black", lw=1)
        ax.invert_yaxis()

        fig.suptitle("Per-Target Analysis: Not All Targets Are Created Equal", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig7_per_target.png")

    # ------------------------------------------------------------------
    # Figure 8: Feature scatter matrix for top features vs pKD
    # ------------------------------------------------------------------

    def pkd_scatters(self, analysis_df, corr_df, n_features=6):
        top = corr_df.head(n_features)
        ncols = 3
        nrows = (n_features + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten()
        pkd_data = analysis_df[analysis_df["pkd"].notna()]

        for i, (_, row) in enumerate(top.iterrows()):
            ax = axes[i]
            feat = row["feature"]
            sub = pkd_data[[feat, "pkd"]].dropna()
            ax.scatter(sub[feat], sub["pkd"], s=20, alpha=0.4, c=PALETTE["primary"],
                       edgecolor="white", linewidth=0.3)
            # Regression line
            z = np.polyfit(sub[feat], sub["pkd"], 1)
            p = np.poly1d(z)
            xline = np.linspace(sub[feat].min(), sub[feat].max(), 100)
            ax.plot(xline, p(xline), "--", color=PALETTE["nonbinder"], lw=2,
                    label=f"r = {row['spearman_r']:.3f}")
            ax.set_xlabel(feat[:35], fontsize=9)
            ax.set_ylabel("pKD")
            ax.set_title(feat[:35], fontsize=10)
            ax.legend(fontsize=8, loc="best")
            ax.text(0.05, 0.95, f"n = {row['n_samples']}\np = {row['p_value']:.1e}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6))

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Computational Features vs Binding Strength (pKD)", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig8_pkd_scatters.png")

    # ------------------------------------------------------------------
    # Figure 9: Correlation heatmap of top features
    # ------------------------------------------------------------------

    def correlation_heatmap(self, analysis_df, features, n_features=15):
        top = features[:n_features]
        valid = [f for f in top if f in analysis_df.columns]
        corr = analysis_df[valid].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                    cbar_kws={"label": "Pearson correlation"})
        ax.set_title("Feature Correlation Matrix (Top Predictors)")
        plt.tight_layout()
        return self._savefig(fig, "fig9_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # Figure 10: Design method comparison
    # ------------------------------------------------------------------

    def design_method_comparison(self, analysis_df):
        bd = analysis_df[analysis_df["binding"].notna()].copy()
        if "design_method" not in bd.columns or bd["design_method"].nunique() < 2:
            return None

        rates = bd.groupby("design_method")["binding"].agg(["mean", "count", "sum"])
        rates = rates[rates["count"] >= 10].sort_values("mean", ascending=False)
        rates["rate"] = rates["mean"] * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(rates)))
        ax.barh(range(len(rates)), rates["count"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([m[:30] for m in rates.index], fontsize=9)
        ax.set_xlabel("Number of designs tested")
        ax.set_title("A. Designs per method")
        ax.invert_yaxis()

        ax = axes[1]
        cmap = ["#2ecc71" if r > 50 else "#e67e22" if r > 25 else "#e74c3c" for r in rates["rate"]]
        ax.barh(range(len(rates)), rates["rate"], color=cmap, edgecolor="white")
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([""] * len(rates))
        ax.set_xlabel("Binding success rate (%)")
        ax.set_title("B. Success rate by design method")
        ax.axvline(50, color="gray", ls="--", alpha=0.5)
        ax.invert_yaxis()

        fig.suptitle("Design Method Comparison", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig10_design_methods.png")

    # ------------------------------------------------------------------
    # Figure 11: Greedy selection progress
    # ------------------------------------------------------------------

    def greedy_progress(self, selector):
        if not selector.scores:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = range(1, len(selector.scores) + 1)
        ax.plot(steps, selector.scores, "o-", color=PALETTE["primary"], lw=2, markersize=8)
        for i, (feat, score) in enumerate(zip(selector.selected, selector.scores)):
            ax.annotate(feat[:25], (i + 1, score), fontsize=7, rotation=20,
                        xytext=(5, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        ax.set_xlabel("Number of features")
        ax.set_ylabel("Average Precision (cross-validated)")
        ax.set_title("Greedy Forward Feature Selection")
        ax.set_xticks(list(steps))
        return self._savefig(fig, "fig11_greedy_progress.png")

    # ------------------------------------------------------------------
    # Figure 12: Precision-Recall and ROC curves
    # ------------------------------------------------------------------

    def pr_roc_curves(self, y_true, y_proba, label="Model"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PR curve
        ax = axes[0]
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(rec, prec, lw=2, color=PALETTE["primary"], label=f"{label} (AP = {ap:.3f})")
        baseline = y_true.sum() / len(y_true)
        ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"Random ({baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("A. Precision-Recall Curve")
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # ROC curve
        ax = axes[1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, color=PALETTE["primary"], label=f"{label} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("B. ROC Curve")
        ax.legend(fontsize=10)

        fig.suptitle("Model Performance Evaluation", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig12_pr_roc.png")

    # ------------------------------------------------------------------
    # Figure 13: Threshold analysis
    # ------------------------------------------------------------------

    def threshold_analysis(self, y_true, y_proba):
        prec, rec, thr = precision_recall_curve(y_true, y_proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        opt_idx = np.argmax(f1[:-1])
        opt_thr = thr[opt_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.plot(thr, prec[:-1], lw=2, color=PALETTE["primary"], label="Precision")
        ax.plot(thr, rec[:-1], lw=2, color=PALETTE["binder"], label="Recall")
        ax.axvline(opt_thr, color=PALETTE["nonbinder"], ls="--", lw=1.5,
                   label=f"Optimal (F1 = {f1[opt_idx]:.3f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("A. Precision and recall vs threshold")
        ax.legend(fontsize=10)

        ax = axes[1]
        ax.plot(thr, f1[:-1], lw=2, color=PALETTE["accent"])
        ax.axvline(opt_thr, color=PALETTE["nonbinder"], ls="--", lw=1.5,
                   label=f"Optimal = {opt_thr:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("B. F1 score vs threshold")
        ax.legend(fontsize=10)

        fig.suptitle("Threshold Optimization: Balancing Precision and Recall", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig13_threshold.png")

    # ------------------------------------------------------------------
    # Figure 14: Confusion matrix
    # ------------------------------------------------------------------

    def confusion(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-binder", "Binder"],
                    yticklabels=["Non-binder", "Binder"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        return self._savefig(fig, "fig14_confusion.png")

    # ------------------------------------------------------------------
    # Figure 15: Enrichment curve (precision at top-k)
    # ------------------------------------------------------------------

    def enrichment_curve(self, y_true, y_proba):
        order = np.argsort(-y_proba)
        y_sorted = np.array(y_true)[order]
        cumsum = np.cumsum(y_sorted)
        k_values = np.arange(1, len(y_sorted) + 1)
        precision_at_k = cumsum / k_values
        total_pos = y_true.sum()
        recall_at_k = cumsum / total_pos

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.plot(k_values, precision_at_k, lw=2, color=PALETTE["primary"])
        ax.axhline(total_pos / len(y_true), color="gray", ls="--", lw=1,
                   label=f"Random ({total_pos / len(y_true):.3f})")
        ax.set_xlabel("Top-k designs selected")
        ax.set_ylabel("Precision")
        ax.set_title("A. Precision at top-k (enrichment)")
        ax.legend(fontsize=10)
        ax.set_xlim([0, min(500, len(y_true))])

        ax = axes[1]
        ax.plot(k_values, recall_at_k * 100, lw=2, color=PALETTE["binder"])
        ax.set_xlabel("Top-k designs selected")
        ax.set_ylabel("Binders recovered (%)")
        ax.set_title("B. Binder recovery at top-k")
        ax.set_xlim([0, min(500, len(y_true))])

        fig.suptitle("Enrichment Analysis: How Many Binders in Your Top Picks?", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig15_enrichment.png")

    # ------------------------------------------------------------------
    # Figure 16: Single feature AP ranking (Overath et al. style)
    # ------------------------------------------------------------------

    def single_feature_ap_ranking(self, ap_df, top_n=20):
        fig, ax = plt.subplots(figsize=(12, 8))
        df = ap_df.head(top_n).copy().sort_values("AP")
        y = np.arange(len(df))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        bars = ax.barh(y, df["AP"], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f[:45] for f in df["feature"]], fontsize=9)
        ax.set_xlabel("Average Precision (AP)")
        ax.set_title("Single Feature Performance: Average Precision for Binder Prediction")
        baseline = df["n_binders"].iloc[0] / df["n_samples"].iloc[0] if len(df) > 0 else 0.1
        ax.axvline(baseline, color="gray", ls="--", lw=1.5,
                   label=f"Random baseline ({baseline:.3f})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._savefig(fig, "fig16_single_feature_ap.png")

    # ------------------------------------------------------------------
    # Figure 17: Interaction features (Overath et al. style)
    # ------------------------------------------------------------------

    def interaction_feature_comparison(self, base_ap_df, interaction_ap_df, top_n=10):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: best individual features
        ax = axes[0]
        df1 = base_ap_df.head(top_n).copy().sort_values("AP")
        y = np.arange(len(df1))
        ax.barh(y, df1["AP"], color=PALETTE["primary"], alpha=0.8, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([f[:40] for f in df1["feature"]], fontsize=9)
        ax.set_xlabel("Average Precision")
        ax.set_title("A. Best individual features")

        # Right: best interaction features
        ax = axes[1]
        df2 = interaction_ap_df.head(top_n).copy().sort_values("AP")
        y = np.arange(len(df2))
        ax.barh(y, df2["AP"], color=PALETTE["secondary"], alpha=0.8, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([f[:40] for f in df2["feature"]], fontsize=9)
        ax.set_xlabel("Average Precision")
        ax.set_title("B. Best interaction features (pairwise products)")

        fig.suptitle("Individual vs Interaction Features for Binding Prediction", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._savefig(fig, "fig17_interaction_features.png")

    # ------------------------------------------------------------------
    # Figure 18: Per-target AP (Overath et al. Fig 3B style)
    # ------------------------------------------------------------------

    def per_target_ap(self, per_target_df, feature_name):
        fig, ax = plt.subplots(figsize=(10, 7))
        df = per_target_df.sort_values("AP")
        y = np.arange(len(df))

        # Color by number of binders
        norm = plt.Normalize(df["n_binders"].min(), df["n_binders"].max())
        colors = plt.cm.YlOrRd(norm(df["n_binders"]))

        bars = ax.barh(y, df["AP"], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([t[:35] for t in df["target"]], fontsize=9)
        ax.set_xlabel("Average Precision")

        short_name = feature_name[:40]
        ax.set_title(f"Per-Target AP for {short_name}")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, label="Number of binders", pad=0.02)

        plt.tight_layout()
        return self._savefig(fig, "fig18_per_target_ap.png")
