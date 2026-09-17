# CDC Diabetes Risk Prediction

Predicting diabetes/prediabetes risk from self-reported health-survey
indicators (no lab work required), using the CDC's 2015 BRFSS survey data.
Includes full EDA, a Logistic Regression baseline, a Random Forest vs.
XGBoost comparison with a fairness-aware (matched-recall) evaluation, and
SHAP-based model interpretation.

📄 **[Read the full report →](reports/final_report.md)**

## Results at a glance

Final model: **XGBoost** (`n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight≈5.54`)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.318 | 0.760 | 0.449 | 0.811 |
| Random Forest | 0.327 | 0.763 | 0.458 | 0.817 |
| **XGBoost (final)** | 0.320 | **0.775** | 0.453 | 0.815 |

*Metrics at default 0.5 threshold on a held-out 20% test set (45,895 rows). See the [final report](reports/final_report.md#4-results) for the fairer matched-recall comparison used to pick the final model, and for threshold-tuning results.*

![Model comparison](reports/figures/03_model_comparison_default_threshold.png)

## Project structure

```
cdc-diabetes-risk-prediction/
├── README.md                  <- you are here
├── requirements.txt
├── LICENSE
├── data/
│   ├── README.md              <- dataset source, schema, how to regenerate
│   ├── raw/                   <- (gitignored) raw data lands here
│   └── processed/             <- (gitignored) cleaned/cached CSV lands here
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_feature_engineering_and_baseline_model.ipynb
│   ├── 03_model_comparison_and_evaluation.ipynb
│   └── 04_model_interpretation_shap.ipynb
├── src/
│   ├── data_loader.py         <- fetch + clean the dataset
│   ├── features.py            <- train/test split, scaling
│   ├── train.py                <- Logistic Regression / Random Forest / XGBoost training
│   ├── evaluate.py            <- metrics, model comparison, matched-recall comparison
│   └── interpret.py           <- SHAP explainability helpers
├── models/                    <- (gitignored) trained model artifacts land here
└── reports/
    ├── final_report.md        <- full write-up: methodology, results, SHAP findings, conclusions
    └── figures/                <- charts referenced in the report
```

## Dataset

[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
— UCI ML Repository, id 891, derived from the CDC's 2015 BRFSS survey.
253,680 rows × 22 columns raw; 229,474 rows after removing exact duplicates.
Target is `Diabetes_binary` (0 = no diabetes, 1 = prediabetes/diabetes),
imbalanced at roughly 85/15. Full schema in [`data/README.md`](data/README.md).

## How to run this project

```bash
git clone <this-repo-url>
cd cdc-diabetes-risk-prediction
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Option A — walk through the analysis notebook by notebook:
jupyter notebook notebooks/

# Option B — use the reusable pipeline directly in Python:
python -c "
from src.data_loader import load_clean_data
from src.features import make_train_test_split, scale_continuous_features
from src.train import train_all_models
from src.evaluate import compare_models

df = load_clean_data()
X_train, X_test, y_train, y_test = make_train_test_split(df)
X_train_scaled, X_test_scaled, scaler = scale_continuous_features(X_train, X_test)

models = train_all_models(X_train, X_train_scaled, y_train)
preds = {n: m.predict(X_test_scaled if n == 'Logistic Regression' else X_test) for n, m in models.items()}
probs = {n: m.predict_proba(X_test_scaled if n == 'Logistic Regression' else X_test)[:, 1] for n, m in models.items()}

print(compare_models(y_test, preds, probs))
"
```

Notebooks are numbered and meant to be run in order — each one recreates the
same seeded train/test split (`random_state=42`) so results line up across
notebooks 02–04.

## Methodology summary

1. **EDA** (`01`): data quality checks, class balance, feature distributions, correlation with target.
2. **Feature engineering + baseline** (`02`): stratified 80/20 split, `StandardScaler` on continuous features (fit on train only), class-imbalance handling via `class_weight='balanced'`, Logistic Regression baseline.
3. **Model comparison** (`03`): Random Forest and XGBoost (`scale_pos_weight` for imbalance) benchmarked against the baseline; threshold tuning; a **matched-recall comparison** that re-thresholds every model to the same recall before comparing precision/F1, for a fairer head-to-head.
4. **Interpretation** (`04`): SHAP `TreeExplainer` on the final XGBoost model — global importance, direction of effect (beeswarm), individual prediction explanations (waterfall), and feature dependence plots.

Full narrative, results tables, and discussion of limitations: **[reports/final_report.md](reports/final_report.md)**.

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib / seaborn · Jupyter

## License

[MIT](LICENSE)
