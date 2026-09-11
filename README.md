# Diabetes Risk Prediction — CDC Diabetes Health Indicators

Predicting diabetes/prediabetes risk from CDC health survey data using exploratory analysis and machine learning.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Author](#author)

## Project Overview

**Business question:** Can we predict which individuals are at elevated risk of prediabetes or
diabetes based on demographic, lifestyle, and health survey indicators? A model like this could
help a clinic or public health program prioritize outreach and screening for at-risk populations.

> **Note:** This is a risk-flagging tool based on self-reported survey data, not a diagnostic
> tool. Predictions should be interpreted as "elevated risk" rather than "diagnosis."

## Dataset

- **Source:** [UCI Machine Learning Repository — CDC Diabetes Health Indicators (ID 891)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- **Origin:** CDC Behavioral Risk Factor Surveillance System (BRFSS), 2015
- **Size:** 253,680 rows, 21 features, 1 binary target
- **Target variable:** `Diabetes_binary` — 0 = no diabetes, 1 = prediabetes or diabetes
- **Feature categories:**
  - Demographics: age, sex, education, income
  - Lifestyle: smoking, physical activity, fruit/vegetable consumption, alcohol use
  - Health conditions: high blood pressure, high cholesterol, heart disease/stroke history, BMI
  - General/mental health: self-rated general health, mental health days, physical health days
  - Healthcare access: has insurance, could not afford doctor visit
- **Known data quality issues:**
  - Significant class imbalance (far more negative than positive cases)
  - ~24,000 duplicate rows in the raw data, removed during cleaning

## Methods

1. **Data Cleaning** ✅
   - Removed ~24,000 duplicate rows
   - Checked and handled missing values
   - Verified data types and value ranges against the feature dictionary

2. **Exploratory Data Analysis (EDA)** ✅
   - Class balance analysis (confirmed significant imbalance toward the negative class)
   - Univariate distributions of key features (BMI, age, general health, etc.)
   - Bivariate analysis: feature relationships with the target
   - Correlation analysis across all features

3. **Feature Engineering** ✅
   - Stratified train/test split (80/20) performed before any scaling, to avoid data leakage
   - Standard scaling applied to continuous features (BMI, MentHlth, PhysHlth, Age, Education, Income), fit on train only
   - Most features were already binary/ordinal and required no additional encoding
   - *(If completed)* Interaction/binned features tested: [DESCRIBE FEATURES TESTED], impact on
     performance: [IMPROVED / NO CHANGE / DEGRADED — with metric delta]

4. **Modeling** ✅
   - **Baseline:** Logistic Regression with `class_weight='balanced'`
   - **Comparison models:** Random Forest (`class_weight='balanced'`), XGBoost (`scale_pos_weight`)
   - *(If completed)* Additional models tested: [LightGBM / CatBoost — list any added]
   - Class imbalance addressed via class weighting
   - *(If completed)* Resampling comparison: SMOTE vs. class weighting — [SUMMARY OF RESULT]

5. **Evaluation** ✅
   - Precision, recall, F1-score, ROC-AUC compared across all models at the default 0.5 threshold
   - **Matched-recall comparison:** thresholds adjusted per model so all are compared at equal
     recall (~77.5%) — isolates precision differences from threshold effects
   - Confusion matrix and ROC curve analysis, with a focus on minimizing false negatives (missed
     at-risk patients)

6. **Interpretation** 🔄
   - Feature importance rankings (from Logistic Regression coefficients and tree-based importances)
   - SHAP values (global summary, beeswarm, waterfall, and dependence plots) computed for the
     final model: [MODEL NAME]

7. **Deployment** — planned
   - Streamlit app for interactive risk prediction based on user-input health indicators
   - App link (once deployed): [STREAMLIT APP URL]

8. **Write-up** — planned
   - Plain-English summary for non-technical audiences: [LINK TO BLOG POST / REPORT, IF CREATED]

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, numpy (`<2` — see note below) |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn, XGBoost[, LightGBM / CatBoost if added] |
| Interpretation | SHAP |
| Deployment | Streamlit |
| Environment | Jupyter Lab |

> **Note:** `numpy<2` is pinned in `requirements.txt` to avoid a binary-compatibility conflict
> with pandas/numba/SHAP under NumPy 2.x.

## Project Structure

```
diabetes-risk-prediction/
├── README.md
├── requirements.txt
├── data/                                        # raw + processed data (not tracked if large)
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb                    ✅ complete
│   ├── 02_feature_engineering_and_baseline_model.ipynb  ✅ complete
│   ├── 03_model_comparison_and_evaluation.ipynb         ✅ complete
│   └── 04_model_interpretation_shap.ipynb               🔄 in progress
├── src/                                         # reusable functions for cleaning/modeling
├── app/                                         # Streamlit deployment app
└── reports/                                     # figures and write-up assets
```

## How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter lab notebooks/01_data_loading_and_eda.ipynb

# Run the Streamlit app (once built)
streamlit run app/app.py
```

## Key Takeaways

- The dataset is meaningfully imbalanced toward non-diabetic respondents ([X]% vs [Y]%), which
  meant accuracy alone would be a misleading metric — precision, recall, F1, and ROC-AUC were
  used instead throughout this project.
- At default thresholds, Logistic Regression, Random Forest, and XGBoost all performed within a
  narrow band of each other (ROC-AUC 0.811–0.816), suggesting the signal in this dataset is
  largely linear and that model complexity alone offers limited additional lift.
- When compared at a matched recall of ~77.5% (rather than default thresholds), all three models
  converged even further (F1: 0.447–0.453), indicating the models had reached a similar
  predictive ceiling given the available features rather than one architecture being clearly
  superior.
- **Final model selected:** [MODEL NAME], chosen because [REASONING — e.g. "narrow edge in
  precision/F1 at matched recall, plus strong compatibility with SHAP interpretation"].
- **Top predictive features (from SHAP):** [FEATURE 1], [FEATURE 2], [FEATURE 3] — [describe
  direction of effect, e.g. "higher BMI and poor self-rated general health both pushed
  predictions toward higher risk"].
- **Notable interactions/surprises from SHAP dependence plots:** [DESCRIBE ANY INTERACTION
  EFFECTS FOUND, OR STATE "none of particular note" IF NOT APPLICABLE].
- **Practical implication:** [ONE SENTENCE ON WHAT THIS MEANS FOR A REAL-WORLD USE CASE — e.g.
  "A screening tool using this model could flag roughly [X]% of at-risk individuals while
  requiring follow-up on [Y]% of the overall population, a trade-off that would need to be
  validated with a clinical stakeholder before deployment."]

## Model Performance

**At default (0.5) threshold:**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.318 | 0.760 | 0.449 | 0.811 |
| Random Forest | 0.334 | 0.743 | 0.461 | 0.816 |
| XGBoost | 0.320 | 0.775 | 0.453 | 0.815 |
| [LightGBM/CatBoost, if added] | [ ] | [ ] | [ ] | [ ] |

**At matched recall (~77.5%)** — thresholds adjusted per model so recall is held constant,
isolating true precision differences from threshold effects:

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.488 | 0.314 | 0.775 | 0.447 |
| Random Forest | 0.473 | 0.319 | 0.775 | 0.452 |
| XGBoost | 0.500 | 0.320 | 0.775 | 0.453 |
| [LightGBM/CatBoost, if added] | [ ] | [ ] | [ ] | [ ] |

**Selected model: [XGBoost — update if this changes].** [REASONING SUMMARY — see Key Takeaways
above for the full explanation].

## Limitations

- Data is self-reported survey data (BRFSS), not clinical records — subject to recall and
  reporting bias.
- The target combines prediabetes and diabetes into a single class, so the model cannot
  distinguish between the two conditions.
- Model reflects patterns in 2015 U.S. survey data and may not generalize to other populations
  or time periods.
- All three core models converged on a similar performance ceiling (~0.81–0.82 ROC-AUC),
  suggesting that further gains are more likely to come from better features or external data
  than from additional algorithms.

## Next Steps

- [x] Complete EDA and document key findings
- [x] Engineer features and build baseline model
- [x] Train and compare additional models
- [ ] Generate SHAP interpretation plots and fill in Key Takeaways
- [ ] *(Optional)* Test SMOTE resampling and/or LightGBM as additional comparisons
- [ ] Build and deploy Streamlit app
- [ ] Write up plain-English summary for non-technical audiences

## Author

[Your Name]
[LinkedIn URL]
[Portfolio site URL]
