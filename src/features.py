"""
Train/test splitting and feature scaling.

Corresponds to notebook 02_feature_engineering_and_baseline_model.ipynb,
sections 2-3.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import TARGET_COL

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Continuous features that benefit from scaling for distance/gradient-based
# models (Logistic Regression). Tree-based models (Random Forest, XGBoost)
# are trained on the unscaled features.
CONTINUOUS_FEATURES = ["BMI", "MentHlth", "PhysHlth", "Age", "Education", "Income"]


def make_train_test_split(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Stratified 80/20 split, identical across all notebooks (seed=42).

    Splitting BEFORE scaling/resampling avoids data leakage.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def scale_continuous_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a StandardScaler on TRAIN only, apply to both train and test.

    Only the continuous columns in CONTINUOUS_FEATURES are scaled; the
    remaining (binary/ordinal) columns are left untouched. Returns the
    scaled copies plus the fitted scaler (needed to transform new data at
    inference time).
    """
    cols = [c for c in CONTINUOUS_FEATURES if c in X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols] = scaler.fit_transform(X_train[cols])
    X_test_scaled[cols] = scaler.transform(X_test[cols])

    return X_train_scaled, X_test_scaled, scaler


def bmi_category(bmi: float) -> str:
    """Standard CDC BMI category buckets (optional derived feature)."""
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    return "obese"
