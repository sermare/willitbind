"""
Visualization Utilities for Protein Binder Analysis

Creates publication-quality plots for analyzing binder prediction performance,
feature importance, and threshold optimization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    average_precision_score, roc_auc_score
)

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_feature_performance(results_df: pd.DataFrame, top_n: int = 20,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot single feature performance (methodology step 8).

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: feature, median_AP, mean_AP, std_AP
    top_n : int
        Number of top features to display
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Select top N features
    plot_data = results_df.head(top_n).copy()

    # Sort by median_AP for plotting
    plot_data = plot_data.sort_values('median_AP')

    # Create barplot
    y_pos = np.arange(len(plot_data))

    ax.barh(y_pos, plot_data['median_AP'], xerr=plot_data['std_AP'],
            color='steelblue', alpha=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data['feature'], fontsize=9)
    ax.set_xlabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features by Performance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  label: str = 'Model',
                                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    label : str
        Label for the curve
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    ax.plot(recall, precision, linewidth=2, label=f'{label} (AP={ap:.3f})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)

    # Add baseline
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline (prevalence={baseline:.3f})')

    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    label: str = 'Model',
                    figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    label : str
        Label for the curve
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           figsize: Tuple[int, int] = (6, 5)) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-binder', 'Binder'],
                yticklabels=['Non-binder', 'Binder'])

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_threshold_analysis(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot precision, recall, and F1 score vs threshold.

    Methodology step 11: Threshold strategy testing.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Precision and Recall vs Threshold
    ax1.plot(thresholds, precision[:-1], linewidth=2, label='Precision', color='blue')
    ax1.plot(thresholds, recall[:-1], linewidth=2, label='Recall', color='green')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'Optimal (F1={f1_scores[optimal_idx]:.3f})')

    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Precision and Recall vs Threshold', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: F1 Score vs Threshold
    ax2.plot(thresholds, f1_scores[:-1], linewidth=2, color='purple')
    ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'Optimal={optimal_threshold:.3f}')
    ax2.axhline(f1_scores[optimal_idx], color='red', linestyle=':', linewidth=1,
                alpha=0.5)

    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot feature importance from model coefficients.

    Methodology step 15: Feature importance analysis.

    Parameters
    ----------
    feature_names : List[str]
        Feature names
    importances : np.ndarray
        Feature importance values (e.g., coefficients)
    top_n : int
        Number of top features to display
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create DataFrame and sort by absolute importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'abs_importance': np.abs(importances)
    })

    importance_df = importance_df.sort_values('abs_importance', ascending=False).head(top_n)
    importance_df = importance_df.sort_values('importance')

    # Create barplot
    colors = ['red' if x < 0 else 'green' for x in importance_df['importance']]
    y_pos = np.arange(len(importance_df))

    ax.barh(y_pos, importance_df['importance'], color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'], fontsize=9)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cross_validation_results(cv_results: Dict[str, List[float]],
                                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot cross-validation results across folds.

    Parameters
    ----------
    cv_results : Dict[str, List[float]]
        Dictionary with metric names as keys and list of scores as values
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    metric_names = list(cv_results.keys())
    n_metrics = len(metric_names)

    x = np.arange(len(cv_results[metric_names[0]]))  # Fold indices
    width = 0.8 / n_metrics

    for i, metric in enumerate(metric_names):
        offset = (i - n_metrics/2) * width + width/2
        ax.bar(x + offset, cv_results[metric], width, label=metric, alpha=0.8)

    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in x])
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_per_target_performance(target_results: pd.DataFrame,
                                  metric: str = 'AP',
                                  figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot performance across different targets.

    Methodology step 10: Per-target performance evaluation.

    Parameters
    ----------
    target_results : pd.DataFrame
        DataFrame with columns: target, AP (or other metrics)
    metric : str
        Metric name to plot
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by metric
    plot_data = target_results.sort_values(metric)

    x_pos = np.arange(len(plot_data))

    ax.bar(x_pos, plot_data[metric], color='steelblue', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_data['target'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Across Targets ({metric})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add median line
    median_value = plot_data[metric].median()
    ax.axhline(median_value, color='red', linestyle='--', linewidth=1.5,
               label=f'Median={median_value:.3f}')
    ax.legend(loc='best')

    plt.tight_layout()
    return fig


def plot_filtering_strategy_comparison(strategy_results: pd.DataFrame,
                                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compare different filtering strategies.

    Methodology step 18: Filtering strategy validation.

    Parameters
    ----------
    strategy_results : pd.DataFrame
        DataFrame with columns: k, strategy, precision, recall
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    strategies = strategy_results['strategy'].unique()

    # Plot 1: Precision vs k
    for strategy in strategies:
        data = strategy_results[strategy_results['strategy'] == strategy]
        ax1.plot(data['k'], data['precision'], marker='o', linewidth=2,
                 label=strategy, alpha=0.8)

    ax1.set_xlabel('Top-k Selected', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision vs Top-k', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Recall vs k
    for strategy in strategies:
        data = strategy_results[strategy_results['strategy'] == strategy]
        ax2.plot(data['k'], data['recall'], marker='o', linewidth=2,
                 label=strategy, alpha=0.8)

    ax2.set_xlabel('Top-k Selected', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Binder Recovery', fontsize=12, fontweight='bold')
    ax2.set_title('Binder Recovery vs Top-k', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(data: pd.DataFrame, features: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot correlation heatmap of features.

    Parameters
    ----------
    data : pd.DataFrame
        Feature data
    features : List[str], optional
        Specific features to include
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if features is not None:
        plot_data = data[features]
    else:
        plot_data = data.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = plot_data.corr()

    # Plot heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})

    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig
