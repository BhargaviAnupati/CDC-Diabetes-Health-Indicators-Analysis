"""
Model training: Logistic Regression baseline, Random Forest, XGBoost.

Corresponds to notebook 02 (baseline) and notebook 03 (Random Forest,
XGBoost, and the model comparison).

All three models handle the ~85/15 class imbalance directly rather than via
resampling:
  - Logistic Regression / Random Forest -> class_weight='balanced'
  - XGBoost -> scale_pos_weight = (negative count / positive count)
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features import RANDOM_STATE


def train_logistic_regression(X_train_scaled: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Baseline model (notebook 02). Requires scaled continuous features."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Random Forest (notebook 03). Trained on UNSCALED features -- tree
    splits are invariant to monotonic feature scaling."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """XGBoost (notebook 03), also the model retrained for SHAP in notebook 04.

    Trained on UNSCALED features.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, X_train_scaled, y_train):
    """Convenience wrapper: train all three models used in the project.

    Returns a dict of {name: fitted_model}.
    """
    return {
        "Logistic Regression": train_logistic_regression(X_train_scaled, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }
