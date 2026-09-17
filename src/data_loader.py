"""
Data loading utilities for the CDC Diabetes Health Indicators dataset.

Source: UCI Machine Learning Repository, dataset id 891
(CDC Diabetes Health Indicators, derived from the 2015 BRFSS survey).
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

Corresponds to notebook 01_data_loading_and_eda.ipynb, section 1-2.
"""

from __future__ import annotations

import pandas as pd

TARGET_COL = "Diabetes_binary"
UCI_DATASET_ID = 891


def fetch_raw_data() -> pd.DataFrame:
    """Download the raw CDC Diabetes Health Indicators dataset via ucimlrepo.

    Returns the features and target concatenated into a single DataFrame,
    exactly as loaded in notebook 01 (253,680 rows x 22 columns, no
    duplicates dropped yet).
    """
    from ucimlrepo import fetch_ucirepo

    cdc_diabetes = fetch_ucirepo(id=UCI_DATASET_ID)
    X = cdc_diabetes.data.features
    y = cdc_diabetes.data.targets
    df = pd.concat([X, y], axis=1)
    return df


def load_clean_data(cache_path: str | None = "data/processed/cdc_diabetes_clean.csv") -> pd.DataFrame:
    """Load the dataset and apply the cleaning step used throughout the project.

    The only cleaning step identified during EDA was removing exact duplicate
    rows (24,206 rows / 9.5% of the raw data were duplicates). No missing
    values were found in any column.

    Parameters
    ----------
    cache_path:
        If given and the file already exists, load from this CSV instead of
        re-downloading. If given and the file does NOT exist, the cleaned
        data is fetched fresh and written there for next time. Pass None to
        always fetch fresh without caching.

    Returns
    -------
    DataFrame with shape (229474, 22), no duplicate rows.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    df = fetch_raw_data()
    df_clean = df.drop_duplicates().reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_clean.to_csv(cache_path, index=False)

    return df_clean


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Quick data-quality snapshot, matching the checks run in notebook 01."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    data = load_clean_data(cache_path=None)
    print(data_quality_summary(data))
    print(data[TARGET_COL].value_counts(normalize=True).round(3))
