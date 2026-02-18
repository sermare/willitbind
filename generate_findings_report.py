#!/usr/bin/env python3
"""
Generate comprehensive findings report from computational features analysis.
Focuses on statistical significance, per-target analysis, and evidence-based recommendations.
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
import json
from datetime import datetime

sys.path.insert(0, '.')

from src.utils.data_loader import ProteinDataLoader


def extract_binding_by_target(evaluations_str):
    """Extract experimental binding data organized by target."""
    try:
        evals = json.loads(evaluations_str) if isinstance(evaluations_str, str) else []
    except:
        return {}

    targets = {}

    for item in evals:
        if item.get('type') != 'experimental':
            continue

        target = item.get('target')
        if not target or target == 'unknown':
            continue

        metric = item.get('metric')
        value = item.get('value')

        if target not in targets:
            targets[target] = {'binding': None, 'kd': None, 'pkd': None}

        if metric == 'binding' and isinstance(value, bool):
            targets[target]['binding'] = int(value)
        elif metric == 'kd' and isinstance(value, (int, float)) and value > 0:
            targets[target]['kd'] = value
            targets[target]['pkd'] = -np.log10(value)

    return targets


def compute_statistical_tests(data, feature_cols, target_col='binding'):
    """Compute statistical tests for all features."""
    results = []

    for feature in feature_cols:
        feature_data = data[[feature, target_col]].dropna()

        if len(feature_data) < 10:
            continue

        binders = feature_data[feature_data[target_col] == 1][feature]
        non_binders = feature_data[feature_data[target_col] == 0][feature]

        if len(binders) < 5 or len(non_binders) < 5:
            continue

        # Mann-Whitney U test
        statistic, p_value = mannwhitneyu(binders, non_binders, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((binders.std()**2 + non_binders.std()**2) / 2)
        effect_size = (binders.mean() - non_binders.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': feature,
            'binder_mean': binders.mean(),
            'binder_std': binders.std(),
            'nonbinder_mean': non_binders.mean(),
            'nonbinder_std': non_binders.std(),
            'difference': binders.mean() - non_binders.mean(),
            'effect_size': effect_size,
            'p_value': p_value,
            'n_binders': len(binders),
            'n_nonbinders': len(non_binders)
        })

    return pd.DataFrame(results).sort_values('p_value') if results else pd.DataFrame()


def compute_correlations(data, target_col, feature_cols):
    """Compute Spearman correlations with target."""
    results = []

    valid_data = data[data[target_col].notna()].copy()

    if len(valid_data) < 10:
        return pd.DataFrame()

    for feature in feature_cols:
        valid_pairs = valid_data[[feature, target_col]].dropna()

        if len(valid_pairs) < 10:
            continue

        x = valid_pairs[feature].values
        y = valid_pairs[target_col].values

        if x.std() == 0 or y.std() == 0:
            continue

        corr, p_value = spearmanr(x, y)

        results.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value,
            'n_samples': len(valid_pairs)
        })

    return pd.DataFrame(results).sort_values('abs_correlation', ascending=False) if results else pd.DataFrame()


def generate_report(output_path='FINDINGS_REPORT.md'):
    """Generate comprehensive findings report."""

    print("Loading data...")
    loader = ProteinDataLoader('proteinbase_all_data_28_01_2026.csv')
    loader.load_data()

    # Get computational and experimental features
    computational_df = loader.get_computational_features()
    experimental_df = loader.get_experimental_features()

    # Extract target-specific binding
    raw_data = loader.raw_data.copy()
    raw_data['target_binding'] = raw_data['evaluations'].apply(extract_binding_by_target)

    # Create long-format dataset
    records = []
    for idx, row in raw_data.iterrows():
        for target, binding_data in row['target_binding'].items():
            record = {
                'sample_id': idx,
                'protein_id': row.get('id', idx),
                'target': target,
                **binding_data
            }
            records.append(record)

    binding_df = pd.DataFrame(records)

    # Merge with computational features
    comp_numeric = computational_df.select_dtypes(include=[np.number])
    analysis_df = binding_df.merge(comp_numeric, left_on='sample_id', right_index=True, how='left')

    # Get usable features
    comp_features = [col for col in comp_numeric.columns if col in analysis_df.columns]
    feature_completeness = pd.DataFrame({
        'feature': comp_features,
        'n_available': [analysis_df[col].notna().sum() for col in comp_features],
        'pct_available': [analysis_df[col].notna().sum() / len(analysis_df) * 100 for col in comp_features]
    }).sort_values('pct_available', ascending=False)

    usable_features = feature_completeness[feature_completeness['pct_available'] >= 10]['feature'].tolist()

    print(f"Found {len(usable_features)} usable computational features")
    print(f"Analyzing {len(analysis_df)} protein-target pairs")
    print(f"Unique targets: {analysis_df['target'].nunique()}")

    # Overall analysis
    print("\nComputing overall statistical tests...")
    binding_data = analysis_df[analysis_df['binding'].notna()].copy()
    overall_stats = compute_statistical_tests(binding_data, usable_features, 'binding')

    # pKD correlations
    print("Computing pKD correlations...")
    pkd_corr = compute_correlations(analysis_df, 'pkd', usable_features)

    # Per-target analysis
    print("\nAnalyzing per target...")
    target_results = {}
    targets = analysis_df['target'].unique()

    for target in targets:
        target_data = analysis_df[analysis_df['target'] == target]

        # Binary binding
        if target_data['binding'].notna().sum() >= 10:
            target_stats = compute_statistical_tests(target_data, usable_features, 'binding')
            target_results[target] = {
                'stats': target_stats,
                'n_samples': len(target_data),
                'n_binders': (target_data['binding'] == 1).sum(),
                'n_nonbinders': (target_data['binding'] == 0).sum()
            }

        # pKD correlation
        if target_data['pkd'].notna().sum() >= 10:
            target_pkd = compute_correlations(target_data, 'pkd', usable_features)
            if target not in target_results:
                target_results[target] = {'n_samples': len(target_data)}
            target_results[target]['pkd_corr'] = target_pkd
            target_results[target]['pkd_mean'] = target_data['pkd'].mean()
            target_results[target]['pkd_std'] = target_data['pkd'].std()

    # Focus on Boltz-related targets
    boltz_targets = [t for t in targets if 'boltz' in t.lower()]

    # Generate report
    print(f"\nGenerating report to {output_path}...")

    with open(output_path, 'w') as f:
        # Header
        f.write("# Computational Features Analysis: Comprehensive Findings Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Analysis Scope:** Computational features predicting experimental binding outcomes\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total protein-target pairs analyzed:** {len(analysis_df):,}\n")
        f.write(f"- **Unique targets:** {len(targets)}\n")
        f.write(f"- **Computational features analyzed:** {len(usable_features)}\n")
        f.write(f"- **Samples with binding data:** {len(binding_data):,}\n")
        f.write(f"- **Samples with KD data:** {analysis_df['kd'].notna().sum():,}\n")

        if len(overall_stats) > 0:
            sig_features = overall_stats[overall_stats['p_value'] < 0.05]
            f.write(f"- **Statistically significant features (p<0.05):** {len(sig_features)} / {len(overall_stats)}\n")

        f.write("\n---\n\n")

        # Overall Findings
        f.write("## 1. Overall Statistical Findings\n\n")
        f.write("### 1.1 Top Features by Statistical Significance\n\n")
        f.write("Features that show the strongest discrimination between binders and non-binders across all targets.\n\n")

        if len(overall_stats) > 0:
            f.write("| Rank | Feature | p-value | Effect Size | Binder Mean | Non-binder Mean | Δ |\n")
            f.write("|------|---------|---------|-------------|-------------|-----------------|---|\n")

            for idx, row in overall_stats.head(20).iterrows():
                sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                f.write(f"| {idx+1} | `{row['feature'][:50]}` | {row['p_value']:.2e} {sig} | "
                       f"{row['effect_size']:+.3f} | {row['binder_mean']:.3f} | "
                       f"{row['nonbinder_mean']:.3f} | {row['difference']:+.3f} |\n")

            f.write("\n**Significance levels:** *** p<0.001, ** p<0.01, * p<0.05\n\n")

            # Interpretation
            f.write("### 1.2 Interpretation of Top Features\n\n")

            top5 = overall_stats.head(5)
            for idx, row in top5.iterrows():
                f.write(f"**{idx+1}. {row['feature']}**\n\n")

                direction = "higher" if row['difference'] > 0 else "lower"
                magnitude = abs(row['difference'])

                f.write(f"- **Effect:** Binders show {direction} values ({row['difference']:+.3f} units)\n")
                f.write(f"- **Statistical significance:** p = {row['p_value']:.2e}\n")
                f.write(f"- **Effect size:** {row['effect_size']:.3f} (")

                if abs(row['effect_size']) < 0.2:
                    f.write("negligible)\n")
                elif abs(row['effect_size']) < 0.5:
                    f.write("small)\n")
                elif abs(row['effect_size']) < 0.8:
                    f.write("medium)\n")
                else:
                    f.write("large)\n")

                f.write(f"- **Sample sizes:** {row['n_binders']} binders, {row['n_nonbinders']} non-binders\n\n")

        # pKD correlations
        f.write("### 1.3 Features Correlated with Binding Strength (pKD)\n\n")
        f.write("Features that correlate with binding affinity (higher pKD = stronger binding).\n\n")

        if len(pkd_corr) > 0:
            f.write("| Rank | Feature | Correlation | p-value | Samples |\n")
            f.write("|------|---------|-------------|---------|----------|\n")

            for idx, row in pkd_corr.head(15).iterrows():
                sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                f.write(f"| {idx+1} | `{row['feature'][:50]}` | {row['correlation']:+.3f} | "
                       f"{row['p_value']:.2e} {sig} | {row['n_samples']} |\n")

            f.write("\n**Positive correlation:** Higher feature value → Stronger binding (lower KD)\n")
            f.write("**Negative correlation:** Higher feature value → Weaker binding (higher KD)\n\n")

        f.write("\n---\n\n")

        # Per-Target Analysis
        f.write("## 2. Per-Target Analysis\n\n")
        f.write("Different targets may exhibit different feature importance patterns.\n\n")

        # Summary table
        f.write("### 2.1 Target Overview\n\n")
        f.write("| Target | Samples | Binders | Non-binders | pKD Available | Mean pKD |\n")
        f.write("|--------|---------|---------|-------------|---------------|----------|\n")

        for target in sorted(target_results.keys(), key=lambda x: target_results[x]['n_samples'], reverse=True):
            result = target_results[target]
            n_bind = result.get('n_binders', 'N/A')
            n_nonbind = result.get('n_nonbinders', 'N/A')
            pkd_mean = f"{result['pkd_mean']:.2f} ± {result['pkd_std']:.2f}" if 'pkd_mean' in result else 'N/A'
            pkd_avail = result.get('pkd_corr', pd.DataFrame()).shape[0] if 'pkd_corr' in result else 0

            f.write(f"| `{target[:40]}` | {result['n_samples']} | {n_bind} | {n_nonbind} | "
                   f"{pkd_avail} | {pkd_mean} |\n")

        f.write("\n")

        # Detailed per-target findings
        f.write("### 2.2 Detailed Target-Specific Findings\n\n")

        # Sort by sample size
        sorted_targets = sorted(target_results.items(),
                               key=lambda x: x[1]['n_samples'],
                               reverse=True)

        for target, result in sorted_targets[:10]:  # Top 10 targets by sample size
            f.write(f"#### {target}\n\n")
            f.write(f"**Sample size:** {result['n_samples']}\n\n")

            if 'stats' in result and len(result['stats']) > 0:
                f.write("**Top 5 discriminative features (binary binding):**\n\n")
                f.write("| Feature | p-value | Effect Size | Binder Mean | Non-binder Mean |\n")
                f.write("|---------|---------|-------------|-------------|------------------|\n")

                for idx, row in result['stats'].head(5).iterrows():
                    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                    f.write(f"| `{row['feature'][:40]}` | {row['p_value']:.2e} {sig} | "
                           f"{row['effect_size']:+.3f} | {row['binder_mean']:.3f} | "
                           f"{row['nonbinder_mean']:.3f} |\n")
                f.write("\n")

            if 'pkd_corr' in result and len(result['pkd_corr']) > 0:
                f.write("**Top 5 features correlated with binding strength:**\n\n")
                f.write("| Feature | Correlation | p-value |\n")
                f.write("|---------|-------------|---------|\n")

                for idx, row in result['pkd_corr'].head(5).iterrows():
                    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                    f.write(f"| `{row['feature'][:40]}` | {row['correlation']:+.3f} | {row['p_value']:.2e} {sig} |\n")
                f.write("\n")

            f.write("\n")

        f.write("\n---\n\n")

        # Boltz-specific analysis
        if boltz_targets:
            f.write("## 3. Boltz-Related Targets (Special Focus)\n\n")
            f.write(f"Found {len(boltz_targets)} Boltz-related targets:\n\n")

            for target in boltz_targets:
                if target in target_results:
                    result = target_results[target]
                    f.write(f"### {target}\n\n")
                    f.write(f"- **Samples:** {result['n_samples']}\n")

                    if 'stats' in result and len(result['stats']) > 0:
                        top_feature = result['stats'].iloc[0]
                        f.write(f"- **Top discriminative feature:** `{top_feature['feature']}`\n")
                        f.write(f"  - p-value: {top_feature['p_value']:.2e}\n")
                        f.write(f"  - Effect size: {top_feature['effect_size']:+.3f}\n")

                    if 'pkd_corr' in result and len(result['pkd_corr']) > 0:
                        top_corr = result['pkd_corr'].iloc[0]
                        f.write(f"- **Top strength predictor:** `{top_corr['feature']}`\n")
                        f.write(f"  - Correlation: {top_corr['correlation']:+.3f}\n")
                        f.write(f"  - p-value: {top_corr['p_value']:.2e}\n")

                    f.write("\n")
        else:
            f.write("## 3. Boltz-Related Targets\n\n")
            f.write("No Boltz-specific targets identified in dataset.\n\n")

        f.write("\n---\n\n")

        # Recommendations
        f.write("## 4. Evidence-Based Recommendations\n\n")

        f.write("### 4.1 Best Overall Features for Binding Prediction\n\n")

        if len(overall_stats) > 0:
            f.write("Based on statistical significance and effect sizes, the top recommended features are:\n\n")

            for idx, row in overall_stats.head(5).iterrows():
                f.write(f"{idx+1}. **{row['feature']}**\n")
                f.write(f"   - Rationale: p={row['p_value']:.2e}, effect size={row['effect_size']:.3f}\n")
                f.write(f"   - Use: {'Higher values favor binding' if row['difference'] > 0 else 'Lower values favor binding'}\n\n")

        f.write("### 4.2 Best Features for Binding Strength Prediction\n\n")

        if len(pkd_corr) > 0:
            f.write("For predicting binding affinity (KD), prioritize:\n\n")

            for idx, row in pkd_corr.head(5).iterrows():
                direction = "stronger" if row['correlation'] > 0 else "weaker"
                f.write(f"{idx+1}. **{row['feature']}**\n")
                f.write(f"   - Correlation with pKD: {row['correlation']:+.3f} (p={row['p_value']:.2e})\n")
                f.write(f"   - Interpretation: Higher values associate with {direction} binding\n\n")

        f.write("### 4.3 Target-Specific Recommendations\n\n")

        f.write("Certain features show strong target-specific patterns. Consider:\n\n")
        f.write("- Using **target-specific models** rather than global models\n")
        f.write("- Features that work well across multiple targets are more generalizable\n")
        f.write("- Some targets may require different computational metrics\n\n")

        # Feature consistency across targets
        if len(target_results) >= 3:
            f.write("**Feature consistency analysis:**\n\n")

            feature_counts = {}
            for target, result in target_results.items():
                if 'stats' in result and len(result['stats']) > 0:
                    top_feature = result['stats'].iloc[0]['feature']
                    feature_counts[top_feature] = feature_counts.get(top_feature, 0) + 1

            if feature_counts:
                sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                f.write("Features appearing as top predictor across multiple targets:\n\n")
                for feature, count in sorted_features[:5]:
                    f.write(f"- `{feature}`: {count} targets\n")
                f.write("\n")

        f.write("### 4.4 Model Development Strategy\n\n")
        f.write("**Recommended approach:**\n\n")
        f.write("1. **Baseline model:** Use top 3-5 overall significant features\n")
        f.write("2. **Enhanced model:** Add target-specific features as needed\n")
        f.write("3. **Regularization:** Use Lasso/Ridge to prevent overfitting\n")
        f.write("4. **Validation:** Cross-validate on held-out targets, not just samples\n")
        f.write("5. **Threshold optimization:** Adjust based on use case (screening vs validation)\n\n")

        f.write("\n---\n\n")

        # Methodology
        f.write("## 5. Statistical Methodology\n\n")
        f.write("### Tests Performed\n\n")
        f.write("- **Mann-Whitney U test:** Non-parametric test for comparing binders vs non-binders\n")
        f.write("- **Spearman correlation:** Rank-based correlation with binding strength (pKD)\n")
        f.write("- **Effect size (Cohen's d):** Standardized measure of difference magnitude\n")
        f.write("- **Significance threshold:** p < 0.05 (with Bonferroni correction recommended for multiple testing)\n\n")

        f.write("### Data Quality Notes\n\n")
        f.write(f"- Only features with ≥10% data availability included ({len(usable_features)} features)\n")
        f.write("- Only targets with ≥10 samples analyzed in detail\n")
        f.write("- Missing values handled by exclusion on per-test basis\n")
        f.write("- Computational features only (no experimental data leakage)\n\n")

        f.write("\n---\n\n")

        # Appendix
        f.write("## Appendix A: All Statistically Significant Features\n\n")

        if len(overall_stats) > 0:
            sig_features = overall_stats[overall_stats['p_value'] < 0.05]
            f.write(f"Total significant features at p<0.05: {len(sig_features)}\n\n")

            if len(sig_features) > 0:
                f.write("| Feature | p-value | Effect Size | Difference |\n")
                f.write("|---------|---------|-------------|------------|\n")

                for idx, row in sig_features.iterrows():
                    f.write(f"| `{row['feature'][:60]}` | {row['p_value']:.2e} | "
                           f"{row['effect_size']:+.3f} | {row['difference']:+.3f} |\n")

        f.write("\n---\n\n")
        f.write("**End of Report**\n")

    print(f"\n✓ Report generated: {output_path}")
    print(f"\n{len(overall_stats)} features analyzed")
    print(f"{len([t for t in target_results if 'stats' in target_results[t]])} targets with detailed analysis")


if __name__ == "__main__":
    generate_report()
