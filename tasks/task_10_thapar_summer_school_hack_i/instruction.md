# Regression: Stock Price Prediction (Thapar Summer School 2025 - Hack-I)

Predict the target value based on various features using regression modeling.

Training data: /app/data/train.csv
Test data: Hidden from you
Output: /app/predict.py that loads your trained model and makes predictions

## Task

1. Train a regression model to predict target values
2. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Outputs predictions to /app/predictions.csv
   - Columns: id,target (target must be numeric values)
   - Format should match: id,target

## Data Format

- Training CSV: id, feature_1, feature_2, ..., feature_N, target (target is the target variable)
- Test CSV: id, feature_1, feature_2, ..., feature_N (no target column)
- Features may include numeric and categorical variables
- Target is a continuous numeric value

## Evaluation

Your model is evaluated using R² (Coefficient of Determination):
- R² = 1 - (SS_res / SS_tot)
  - SS_res = Σ(predicted_i - actual_i)² (sum of squared residuals)
  - SS_tot = Σ(actual_i - mean(actual))² (total sum of squares)
- R² ranges from -∞ to 1.0 (higher is better)
- R² = 1.0 means perfect predictions
- R² = 0.0 means model performs as well as predicting the mean
- R² < 0.0 means model performs worse than predicting the mean
- **Passing threshold: R² ≥ 0.38** (reasonable threshold for regression tasks)

The evaluation checks:
- All test samples have predictions
- Predictions are valid numeric values
- Output format is correct (id, target)
- No extra predictions for unknown IDs

## Available packages

Python standard library, NumPy, pandas, scikit-learn, lightgbm, xgboost, joblib

## Competition Information

This task is based on the Thapar Summer School 2025 - Hack-I competition:
- **Problem type**: Regression
- **Evaluation metric**: R² (Coefficient of Determination)
- **Submission format**: CSV with columns `id` and `target`

