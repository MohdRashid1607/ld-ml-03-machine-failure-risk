"""
Evaluation metrics for anomaly detection.

Provides precision@K, recall@K, and summary statistics for anomaly scores.
When no ground-truth labels are available, human-reviewed samples should be
used for precision/recall calculations.
"""

import numpy as np
from typing import Optional


def precision_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Precision at K: what fraction of the top-K scored samples are true anomalies?

    Parameters
    ----------
    scores : np.ndarray
        1D array of anomaly scores (higher = more anomalous).
    labels : np.ndarray
        1D array of ground-truth labels (1 = anomaly, 0 = normal).
    k : int
        Number of top-scoring samples to evaluate.

    Returns
    -------
    float
        Precision at K (proportion of true anomalies in top K).
    """
    if k <= 0 or k > len(scores):
        raise ValueError(f"k must be between 1 and {len(scores)}, got {k}.")
    top_k_indices = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_indices].sum()) / k


def recall_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Recall at K: what fraction of all true anomalies appear in the top-K scores?

    Parameters
    ----------
    scores : np.ndarray
        1D array of anomaly scores (higher = more anomalous).
    labels : np.ndarray
        1D array of ground-truth labels (1 = anomaly, 0 = normal).
    k : int
        Number of top-scoring samples to evaluate.

    Returns
    -------
    float
        Recall at K. Returns 0.0 if no true anomalies exist.
    """
    if k <= 0 or k > len(scores):
        raise ValueError(f"k must be between 1 and {len(scores)}, got {k}.")
    total_positives = labels.sum()
    if total_positives == 0:
        return 0.0
    top_k_indices = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_indices].sum()) / total_positives


def anomaly_summary(scores: np.ndarray, threshold: float) -> dict:
    """
    Summarize anomaly score distribution and flagged counts.

    Parameters
    ----------
    scores : np.ndarray
        1D array of anomaly scores.
    threshold : float
        Score above which samples are flagged as anomalous.

    Returns
    -------
    dict
        Summary statistics including count, mean, std, min, max, median,
        p95, n_flagged, and flag_rate.
    """
    n_flagged = int((scores > threshold).sum())
    return {
        "n_samples": int(len(scores)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "median_score": float(np.median(scores)),
        "p95_score": float(np.percentile(scores, 95)),
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "flag_rate": float(n_flagged / len(scores)) if len(scores) > 0 else 0.0,
    }
