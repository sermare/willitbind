#!/usr/bin/env python3
"""
WillItBind: Full analysis pipeline.

Runs the complete analysis on the 5,253 protein binder dataset, extending the
methodology of Overath et al. (2025) "Predicting Experimental Success in De Novo
Binder Design" from 3,766 to 5,253 designs across 24 targets.

Pipeline steps:
 1. Load and parse the 5k+ design dataset
 2. Dataset overview visualization
 3. Statistical comparison of binders vs non-binders (Mann-Whitney U)
 4. Single feature Average Precision ranking (as in Overath et al.)
 5. Interaction features: pairwise products (confidence x physicochemical)
 6. Correlation analysis with binding affinity (pKD)
 7. Per-target analysis across 24 targets
 8. Per-target AP for the best single feature
 9. LASSO feature selection (binary + affinity)
10. Greedy forward feature selection with LOGO-CV
11. Model training, evaluation, enrichment curves
12. Feature consistency across targets
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from willitbind.data import BinderDataset
from willitbind.features import FeatureAnalyzer
from willitbind.models import (
    BindingPredictor, GreedySelector, single_feature_performance, optimal_threshold,
)
from willitbind.plots import WillItPlot

DATA_PATH = os.path.join(PROJECT_ROOT, "proteinbase_all_data_28_01_2026.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "results", "tables")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    plotter = WillItPlot(output_dir=FIG_DIR)

    # ====================================================================
    # 1. Load data
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading and parsing the dataset")
    print("=" * 70)

    ds = BinderDataset(DATA_PATH)
    ds.load()
    adf = ds.analysis_dataset()
    features = ds.usable_features(min_availability=0.10)
    print(f"Usable features: {len(features)}")

    adf.to_csv(os.path.join(TABLE_DIR, "analysis_dataset.csv"), index=False)

    # ====================================================================
    # 2. Dataset overview figure
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Dataset overview visualization")
    print("=" * 70)

    path = plotter.dataset_overview(adf, features)
    print(f"  -> {path}")

    # ====================================================================
    # 3. Statistical comparison: binders vs non-binders
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Statistical comparison (Mann-Whitney U)")
    print("=" * 70)

    analyzer = FeatureAnalyzer(adf, features)
    stat_df = analyzer.binder_vs_nonbinder()
    stat_df.to_csv(os.path.join(TABLE_DIR, "binder_vs_nonbinder_stats.csv"), index=False)

    n_sig = (stat_df["p_value"] < 0.05).sum()
    n_large = (stat_df["cohens_d"].abs() > 0.5).sum()
    print(f"  Significant features (p < 0.05): {n_sig}/{len(stat_df)}")
    print(f"  Large effects (|d| > 0.5): {n_large}")
    print("\n  Top 10 by significance:")
    for _, row in stat_df.head(10).iterrows():
        print(f"    {row['feature'][:45]:45s}  d={row['cohens_d']:+.3f}  p={row['p_value']:.2e}")

    path = plotter.effect_sizes(stat_df)
    print(f"  -> {path}")
    path = plotter.volcano(stat_df)
    print(f"  -> {path}")
    path = plotter.top_feature_violins(adf, stat_df, n_features=6)
    print(f"  -> {path}")

    # ====================================================================
    # 4. Single feature AP ranking (Overath et al. methodology)
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Single feature Average Precision ranking")
    print("=" * 70)

    ap_df = analyzer.single_feature_ap()
    ap_df.to_csv(os.path.join(TABLE_DIR, "single_feature_ap.csv"), index=False)

    print(f"\n  Top 10 features by AP:")
    for _, row in ap_df.head(10).iterrows():
        print(f"    {row['feature'][:45]:45s}  AP={row['AP']:.3f}  ({row['direction']})  n={row['n_samples']}")

    path = plotter.single_feature_ap_ranking(ap_df)
    print(f"  -> {path}")

    # ====================================================================
    # 5. Interaction features (pairwise products)
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Interaction features (pairwise products)")
    print("  Following Overath et al.: confidence x physicochemical descriptors")
    print("=" * 70)

    interaction_df = analyzer.interaction_feature_ap(top_n_base=15, top_n_interactions=20)
    interaction_df.to_csv(os.path.join(TABLE_DIR, "interaction_feature_ap.csv"), index=False)

    best_base = ap_df.iloc[0]["AP"] if len(ap_df) > 0 else 0
    best_inter = interaction_df.iloc[0]["AP"] if len(interaction_df) > 0 else 0
    improvement = (best_inter / best_base - 1) * 100 if best_base > 0 else 0

    print(f"\n  Best individual AP:   {best_base:.3f}")
    print(f"  Best interaction AP:  {best_inter:.3f}")
    print(f"  Improvement:          {improvement:+.1f}%")
    print(f"\n  Top 5 interaction features:")
    for _, row in interaction_df.head(5).iterrows():
        print(f"    {row['feature'][:55]:55s}  AP={row['AP']:.3f}")

    path = plotter.interaction_feature_comparison(ap_df, interaction_df)
    print(f"  -> {path}")

    # ====================================================================
    # 6. Correlation with binding affinity (pKD)
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Correlation with pKD (binding strength)")
    print("=" * 70)

    corr_df = analyzer.pkd_correlations()
    corr_df.to_csv(os.path.join(TABLE_DIR, "pkd_correlations.csv"), index=False)

    print(f"\n  Top 10 features by |correlation| with pKD:")
    for _, row in corr_df.head(10).iterrows():
        print(f"    {row['feature'][:45]:45s}  r={row['spearman_r']:+.3f}  p={row['p_value']:.2e}")

    path = plotter.pkd_correlation_bars(corr_df)
    print(f"  -> {path}")
    path = plotter.pkd_scatters(adf, corr_df, n_features=6)
    print(f"  -> {path}")

    # ====================================================================
    # 7. Per-target analysis
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Per-target analysis")
    print("=" * 70)

    target_stats = analyzer.per_target_stats()
    print(f"  Analyzed {len(target_stats)} targets:")
    for t, info in sorted(target_stats.items(), key=lambda x: x[1]["n_samples"], reverse=True)[:8]:
        print(f"    {t[:35]:35s}  n={info['n_samples']:4d}  "
              f"binders={info['n_binders']:3d}  rate={info['binding_rate']:.1f}%")

    path = plotter.per_target_comparison(target_stats)
    print(f"  -> {path}")

    # Save per-target summary
    target_summary = []
    for t, info in target_stats.items():
        row = {"target": t, "n_samples": info["n_samples"],
               "n_binders": info["n_binders"], "n_nonbinders": info["n_nonbinders"],
               "binding_rate": info["binding_rate"]}
        if "pkd_mean" in info:
            row["pkd_mean"] = info["pkd_mean"]
            row["pkd_std"] = info["pkd_std"]
        if "top_binary" in info and len(info["top_binary"]) > 0:
            row["top_feature"] = info["top_binary"].iloc[0]["feature"]
            row["top_effect_size"] = info["top_binary"].iloc[0]["cohens_d"]
            row["top_p_value"] = info["top_binary"].iloc[0]["p_value"]
        target_summary.append(row)
    pd.DataFrame(target_summary).to_csv(os.path.join(TABLE_DIR, "per_target_summary.csv"), index=False)

    # ====================================================================
    # 8. Per-target AP for the best single feature (Overath Fig 3B)
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Per-target AP for best feature (Overath et al. Fig 3B)")
    print("=" * 70)

    best_feat_name = ap_df.iloc[0]["feature"] if len(ap_df) > 0 else features[0]
    per_target_ap_df = analyzer.per_target_ap(best_feat_name)
    per_target_ap_df.to_csv(os.path.join(TABLE_DIR, "per_target_ap_best_feature.csv"), index=False)

    print(f"  Feature: {best_feat_name}")
    for _, row in per_target_ap_df.iterrows():
        print(f"    {row['target'][:35]:35s}  AP={row['AP']:.3f}  "
              f"binders={row['n_binders']:.0f}  rate={row['binding_rate']:.2f}")

    path = plotter.per_target_ap(per_target_ap_df, best_feat_name)
    print(f"  -> {path}")

    # ====================================================================
    # 9. Feature correlation heatmap
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Feature correlation heatmap")
    print("=" * 70)

    path = plotter.correlation_heatmap(adf, features, n_features=15)
    print(f"  -> {path}")

    # ====================================================================
    # 10. Design method comparison
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 10: Design method comparison")
    print("=" * 70)

    path = plotter.design_method_comparison(adf)
    print(f"  -> {path}" if path else "  Not enough design methods to compare")

    # ====================================================================
    # 11. LASSO feature selection
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 11: LASSO feature selection")
    print("=" * 70)

    # Binary
    all_coefs_bin, selected_bin = analyzer.lasso_select_binary()
    all_coefs_bin.to_csv(os.path.join(TABLE_DIR, "lasso_binary_all.csv"), index=False)
    selected_bin.to_csv(os.path.join(TABLE_DIR, "lasso_binary_selected.csv"), index=False)
    path = plotter.lasso_coefficients(all_coefs_bin, selected_bin, task="binary")
    print(f"  -> {path}")

    # pKD
    all_coefs_pkd, selected_pkd = analyzer.lasso_select_pkd()
    all_coefs_pkd.to_csv(os.path.join(TABLE_DIR, "lasso_pkd_all.csv"), index=False)
    selected_pkd.to_csv(os.path.join(TABLE_DIR, "lasso_pkd_selected.csv"), index=False)
    path = plotter.lasso_coefficients(all_coefs_pkd, selected_pkd, task="pkd")
    print(f"  -> {path}")

    # ====================================================================
    # 12. Model training and evaluation
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 12: Model training (L1 logistic regression)")
    print("=" * 70)

    bd = adf[adf["binding"].notna()].copy()
    # Use LASSO-selected features if available, else top AP features
    if len(selected_bin) > 0:
        model_features = selected_bin["feature"].tolist()
        print(f"  Using {len(model_features)} LASSO-selected features")
    else:
        model_features = ap_df.head(5)["feature"].tolist()
        print(f"  LASSO found 0 features; using top-5 AP features instead")

    valid_feats = [f for f in model_features if f in bd.columns]
    X = bd[valid_feats].fillna(bd[valid_feats].median())
    y = bd["binding"].astype(int)

    model = BindingPredictor()
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    print(f"  Model performance:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    proba = model.predict_proba(X)
    thr = metrics["threshold"]
    pred = (proba >= thr).astype(int)

    pd.DataFrame({"true_label": y.values, "predicted_proba": proba, "predicted_label": pred}
                 ).to_csv(os.path.join(TABLE_DIR, "predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(TABLE_DIR, "model_metrics.csv"), index=False)
    model.coefficients().to_csv(os.path.join(TABLE_DIR, "model_coefficients.csv"), index=False)

    path = plotter.pr_roc_curves(y.values, proba, label="LASSO L1 Model")
    print(f"  -> {path}")
    path = plotter.threshold_analysis(y.values, proba)
    print(f"  -> {path}")
    path = plotter.confusion(y.values, pred)
    print(f"  -> {path}")
    path = plotter.enrichment_curve(y.values, proba)
    print(f"  -> {path}")

    # ====================================================================
    # 13. Greedy feature selection with LOGO-CV
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 13: Greedy forward feature selection (LOGO-CV by target)")
    print("=" * 70)

    # Use top AP features as candidates
    top_candidate_feats = ap_df.head(20)["feature"].tolist()
    valid_candidates = [f for f in top_candidate_feats if f in bd.columns]
    X_greedy = bd[valid_candidates].fillna(bd[valid_candidates].median())
    groups = bd["target"] if bd["target"].nunique() > 1 else None

    selector = GreedySelector(early_stop=0.002)
    selected = selector.select(X_greedy, y, groups=groups, max_features=8)

    greedy_df = pd.DataFrame({
        "step": range(1, len(selected) + 1),
        "feature": selected,
        "cumulative_AP": selector.scores,
    })
    greedy_df.to_csv(os.path.join(TABLE_DIR, "greedy_selected_features.csv"), index=False)

    path = plotter.greedy_progress(selector)
    print(f"  -> {path}" if path else "  No progress plot (1 or 0 features)")

    # ====================================================================
    # 14. Feature consistency across targets
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 14: Feature consistency across targets")
    print("=" * 70)

    consistency = analyzer.feature_consistency(target_stats)
    consistency.to_csv(os.path.join(TABLE_DIR, "feature_consistency.csv"), index=False)
    print("  Features appearing in top-5 across multiple targets:")
    for _, row in consistency.head(10).iterrows():
        print(f"    {row['feature'][:45]:45s}  {row['n_targets_top5']} targets")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    n_binders = (adf["binding"] == 1).sum()
    n_total = adf["binding"].notna().sum()

    print(f"""
Dataset:
  {len(ds.raw)} total designs, {adf['target'].nunique()} targets
  {n_binders}/{n_total} binders ({n_binders/n_total*100:.1f}%)

Statistical significance:
  {n_sig} features with p < 0.05
  {n_large} features with large effect (|d| > 0.5)

Best single feature:
  {ap_df.iloc[0]['feature']}  (AP = {ap_df.iloc[0]['AP']:.3f})

Best interaction feature:
  {interaction_df.iloc[0]['feature']}  (AP = {interaction_df.iloc[0]['AP']:.3f})
  Improvement over best single: {improvement:+.1f}%

Model performance:
  AP = {metrics['average_precision']:.3f}, AUC = {metrics['roc_auc']:.3f}, F1 = {metrics['f1']:.3f}

LASSO selected: {len(selected_bin)} features (binary), {len(selected_pkd)} (pKD)
Greedy selected: {len(selected)} features

Figures saved to: {FIG_DIR}/
Tables saved to:  {TABLE_DIR}/
""")


if __name__ == "__main__":
    main()
