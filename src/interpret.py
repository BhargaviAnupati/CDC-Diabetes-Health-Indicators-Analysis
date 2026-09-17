"""
Model interpretation via SHAP (SHapley Additive exPlanations).

Corresponds to notebook 04_model_interpretation_shap.ipynb.

Uses shap.TreeExplainer, which is fast and exact for tree-based models
(XGBoost / Random Forest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 15000
RANDOM_STATE = 42


def compute_shap_values(model, X_test: pd.DataFrame, sample_size: int = DEFAULT_SAMPLE_SIZE):
    """Sample the test set and compute SHAP values for the given tree model.

    Sampling keeps SHAP computation fast on large datasets; increase
    sample_size for more precision if you have time to spare.

    Returns (X_sample, shap_values).
    """
    import shap

    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return X_sample, shap_values


def plot_global_importance(shap_values, max_display: int = 15):
    """Bar plot: mean |SHAP value| per feature (magnitude only)."""
    import shap

    shap.plots.bar(shap_values, max_display=max_display)


def plot_beeswarm(shap_values, max_display: int = 15):
    """Beeswarm plot: shows both importance AND the direction of each
    feature's effect (red = high feature value, blue = low)."""
    import shap

    shap.plots.beeswarm(shap_values, max_display=max_display)


def explain_prediction(model, X_sample: pd.DataFrame, shap_values, index: int, max_display: int = 15):
    """Waterfall plot explaining a single prediction, plus its predicted
    probability."""
    import shap

    proba = model.predict_proba(X_sample)[:, 1][index]
    print(f"Predicted probability of diabetes/prediabetes: {proba:.3f}")
    shap.plots.waterfall(shap_values[index], max_display=max_display)
    return proba


def most_and_least_confident(model, X_sample: pd.DataFrame):
    """Indices of the highest- and lowest-predicted-risk rows in the sample
    -- useful for picking illustrative waterfall examples."""
    probs = model.predict_proba(X_sample)[:, 1]
    return int(np.argmax(probs)), int(np.argmin(probs))


def dependence_plots(shap_values, features=("BMI", "GenHlth", "Age", "HighBP")):
    """Scatter (dependence) plots for the given features, colored by the
    SHAP interaction value."""
    import shap
    import matplotlib.pyplot as plt

    for feature in features:
        shap.plots.scatter(shap_values[:, feature], color=shap_values)
        plt.show()
