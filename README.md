# NHANES Exercise Timing and Metabolic Syndrome

Analysis of NHANES 2011-2014 accelerometry data examining whether the time of day of moderate-to-vigorous physical activity (MVPA) is independently associated with metabolic syndrome after controlling for total MVPA volume.

## Repository Structure
- `code/` — Analysis scripts
  - `06_full_regression.py` — Main logistic regression
  - `04_analysis_v2.py` — Data processing and variable construction
  - `xgboost_shap_analysis.py` — XGBoost + SHAP feature importance
  - `generate_all_figures.py` — Figure generation
  - `generate_real_figures.py` — Updated figures (ROC, calibration)
  - `svyglm_analysis.R` — Survey-weighted sensitivity analysis

## Data
NHANES 2011-2014 accelerometry data is publicly available from:
https://www.cdc.gov/nchs/nhanes/

## License
CC BY 4.0
