# Data

## Source

**CDC Diabetes Health Indicators** — UCI Machine Learning Repository, dataset id **891**.
<https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators>

The dataset is derived from the CDC's 2015 Behavioral Risk Factor Surveillance
System (BRFSS) telephone survey, and is distributed via the `ucimlrepo` Python
package (no manual download or API key needed).

- **Raw size:** 253,680 rows × 22 columns (21 features + 1 target), no missing values.
- **After removing exact duplicate rows:** 229,474 rows (24,206 duplicates, 9.5% of the raw data, were dropped — this is a known characteristic of this dataset).
- **Target:** `Diabetes_binary` — `0` = no diabetes, `1` = prediabetes or diabetes.
- **Class balance:** ~84.7% negative / ~15.3% positive (imbalanced).

## Why the raw data isn't committed to this repo

The cleaned dataset is ~230k rows; committing it (or the raw CSV) would bloat
the repository unnecessarily since it can be re-fetched deterministically in
one line. `data/raw/` and `data/processed/` are gitignored except for a
`.gitkeep` placeholder, so the folder structure is preserved but the data
itself isn't tracked.

To (re)generate a local cleaned copy:

```bash
python -c "from src.data_loader import load_clean_data; load_clean_data()"
```

This downloads the dataset via `ucimlrepo`, drops duplicates, and caches the
result at `data/processed/cdc_diabetes_clean.csv`.

## Column reference

| Column | Type | Description |
|---|---|---|
| `Diabetes_binary` | Target (binary) | 0 = no diabetes, 1 = prediabetes or diabetes |
| `HighBP` | Binary | 0 = no high blood pressure, 1 = high blood pressure |
| `HighChol` | Binary | 0 = no high cholesterol, 1 = high cholesterol |
| `CholCheck` | Binary | 1 = had a cholesterol check within 5 years |
| `BMI` | Integer | Body Mass Index |
| `Smoker` | Binary | Smoked ≥100 cigarettes in lifetime |
| `Stroke` | Binary | Ever told they had a stroke |
| `HeartDiseaseorAttack` | Binary | Coronary heart disease (CHD) or myocardial infarction (MI) |
| `PhysActivity` | Binary | Physical activity in past 30 days (excluding job) |
| `Fruits` | Binary | Consumes fruit ≥1 time/day |
| `Veggies` | Binary | Consumes vegetables ≥1 time/day |
| `HvyAlcoholConsump` | Binary | Heavy drinker (adult men >14 drinks/week, women >7 drinks/week) |
| `AnyHealthcare` | Binary | Has any kind of health care coverage |
| `NoDocbcCost` | Binary | Needed to see a doctor in past 12 months but couldn't due to cost |
| `GenHlth` | Ordinal (1–5) | Self-rated general health (1 = excellent, 5 = poor) |
| `MentHlth` | Integer (0–30) | Days of poor mental health in past 30 days |
| `PhysHlth` | Integer (0–30) | Days of poor physical health in past 30 days |
| `DiffWalk` | Binary | Serious difficulty walking or climbing stairs |
| `Sex` | Binary | 0 = female, 1 = male |
| `Age` | Ordinal (1–13) | 13-level age category (see BRFSS codebook, `_AGEG5YR`) |
| `Education` | Ordinal (1–6) | Education level (BRFSS codebook, `EDUCA`) |
| `Income` | Ordinal (1–8) | Income scale (BRFSS codebook, `INCOME2`) |

Full variable definitions are also available programmatically via
`cdc_diabetes.variables` after calling `fetch_ucirepo(id=891)` — see
`notebooks/01_data_loading_and_eda.ipynb`, section 1.
