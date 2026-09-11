# Diabetes Risk Prediction — CDC Diabetes Health Indicators

Predicting diabetes/prediabetes risk from CDC health survey data using exploratory analysis and machine learning.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Next Steps](#next-steps)

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

1. **Data Cleaning**
   - Removed duplicate rows
   - Checked and handled missing values
   - Verified data types and value ranges against the feature dictionary

2. **Exploratory Data Analysis (EDA)**
   - Class balance analysis
   - Univariate distributions of key features (BMI, age, general health, etc.)
   - Bivariate analysis: feature relationships with the target
   - Correlation analysis across all features

3. **Feature Engineering**
   - Encoding of categorical variables where needed
   - Feature scaling for scale-sensitive models
   - (Optional) derived features such as BMI categories or age bins

4. **Modeling**
   - **Baseline:** Logistic Regression (interpretable baseline)
   - **Comparison models:** Random Forest, XGBoost / LightGBM
   - Class imbalance addressed via class weighting and/or resampling (e.g., SMOTE)
   - Hyperparameter tuning via cross-validation

5. **Evaluation**
   - Precision, recall, F1-score, ROC-AUC (accuracy alone is misleading given class imbalance)
   - Confusion matrix analysis with a focus on minimizing false negatives (missed at-risk patients)
   - Model comparison table across all metrics

6. **Interpretation**
   - Feature importance rankings
   - SHAP values to explain individual and global predictions

7. **Deployment**
   - Streamlit app for interactive risk prediction based on user-input health indicators

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn, XGBoost |
| Interpretation | SHAP |
| Deployment | Streamlit |
| Environment | Jupyter Lab |

## Project Structure

```
diabetes-risk-prediction/
├── README.md
├── requirements.txt
├── data/                                  # raw + processed data (not tracked if large)
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_feature_engineering_and_baseline_model.ipynb
│   ├── 03_model_comparison_and_evaluation.ipynb
│   └── 04_model_interpretation_shap.ipynb
├── src/                                   # reusable functions for cleaning/modeling
├── app/                                   # Streamlit deployment app
└── reports/                               # figures and write-up assets
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

## Key Findings

*(Fill in as the project progresses — this is the section most recruiters and interviewers will
actually read. Aim for 3-5 plain-English bullet points, e.g. "Individuals with high blood
pressure and BMI over 30 were X times more likely to be flagged at-risk...")*

- 
- 
- 

## Model Performance

*(Fill in once modeling is complete)*

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |
| XGBoost | | | | |

## Limitations

- Data is self-reported survey data (BRFSS), not clinical records — subject to recall and
  reporting bias.
- The target combines prediabetes and diabetes into a single class, so the model cannot
  distinguish between the two conditions.
- Model reflects patterns in 2015 U.S. survey data and may not generalize to other populations
  or time periods.

## Next Steps

- [ ] Complete EDA and document key findings
- [ ] Engineer features and build baseline model
- [ ] Train and compare additional models
- [ ] Generate SHAP interpretation plots
- [ ] Build and deploy Streamlit app
- [ ] Write up plain-English summary for non-technical audiences

## Author

*(Bhargavi Anupati, LinkedIn, bhargavi-anupati.tech)*
