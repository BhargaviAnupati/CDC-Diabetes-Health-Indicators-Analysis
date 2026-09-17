"""
Evaluation: standard metrics, model comparison table, threshold tuning,
and the matched-recall comparison used to pick the final model.

Corresponds to notebook 03_model_comparison_and_evaluation.ipynb,
sections 5-6b.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)


def get_metrics(name: str, y_true, y_pred, y_proba) -> dict:
    """Metrics for one model at its default (0.5) threshold."""
    return {
        "Model": name,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def compare_models(y_true, predictions: dict, probabilities: dict) -> pd.DataFrame:
    """Build the model comparison table (default 0.5 threshold).

    `predictions` and `probabilities` are {model_name: array} dicts sharing
    the same keys, e.g. {"Logistic Regression": ..., "Random Forest": ...,
    "XGBoost": ...}.
    """
    rows = [
        get_metrics(name, y_true, predictions[name], probabilities[name])
        for name in predictions
    ]
    return pd.DataFrame(rows).set_index("Model").round(3)


def threshold_for_target_recall(y_true, y_proba, target_recall: float) -> float:
    """Highest decision threshold that still achieves >= target_recall.

    A higher threshold at the same recall means fewer false positives
    (better precision) -- this is what makes models comparable when they
    naturally sit at different default-threshold recall levels.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid = recalls[:-1] >= target_recall
    if not valid.any():
        return 0.0
    return thresholds[valid].max()


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred_t = (y_proba >= threshold).astype(int)
    return {
        "Threshold": round(float(threshold), 3),
        "Precision": round(precision_score(y_true, y_pred_t), 3),
        "Recall": round(recall_score(y_true, y_pred_t), 3),
        "F1": round(f1_score(y_true, y_pred_t), 3),
    }


def matched_recall_comparison(y_true, probabilities: dict, target_recall: float) -> pd.DataFrame:
    """Re-threshold every model to the same recall level, then compare
    precision/F1 at that matched point.

    This answers: "if every model had to catch the same share of at-risk
    patients, which one does it with fewer false alarms?" -- a fairer
    comparison than the default-threshold table, since raw thresholds can
    put models at different recall levels by chance.
    """
    rows = {}
    for name, proba in probabilities.items():
        t = threshold_for_target_recall(y_true, proba, target_recall)
        rows[name] = metrics_at_threshold(y_true, proba, t)
    return pd.DataFrame(rows).T


def full_report(y_true, y_pred, target_names=("No Diabetes", "Diabetes/Pre")) -> str:
    return classification_report(y_true, y_pred, target_names=list(target_names))


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
