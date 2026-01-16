# Regression: Yield Prediction

Predict the yield value based on various features using regression modeling.

Training data: /app/data/train.csv
Test data: Hidden from you
Output: /app/predict.py that loads your trained model and makes predictions

## Task

1. Train a regression model to predict yield values
2. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Outputs predictions to /app/predictions.csv
   - Columns: id,yield (yield must be numeric values)
   - Format should match: id,yield

## Data Format

- Training CSV: id, feature_1, feature_2, ..., feature_N, yield (yield is the target variable)
- Test CSV: id, feature_1, feature_2, ..., feature_N (no yield column)
- Features may include numeric and categorical variables
- Target yield is a continuous numeric value

## Evaluation

Your model is evaluated using Mean Absolute Error (MAE):
- MAE = (1/n) * Σ|predicted_i - actual_i|
- Lower MAE is better
- **Passing threshold: MAE ≤ 250.0**

The evaluation checks:
- All test samples have predictions
- Predictions are valid numeric values
- Output format is correct (id, yield)
- No extra predictions for unknown IDs

## Available packages

Python standard library, NumPy, pandas, scikit-learn, lightgbm, xgboost, joblib

