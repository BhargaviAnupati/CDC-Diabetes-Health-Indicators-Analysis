"""
CDC Diabetes Risk Prediction
============================

Reusable, script-friendly versions of the logic developed in the project
notebooks (see ../notebooks). Import these modules from a notebook or a
Python script instead of copy-pasting cells:

    from src.data_loader import load_clean_data
    from src.features import make_train_test_split, scale_continuous_features
    from src.train import train_logistic_regression, train_random_forest, train_xgboost
    from src.evaluate import get_metrics, compare_models, matched_recall_comparison
    from src.interpret import compute_shap_values

Every function here mirrors what actually ran in notebooks 01-04 -- same
random seed (42), same train/test split, same class-imbalance handling --
so results computed through src/ match the reported notebook results.
"""
