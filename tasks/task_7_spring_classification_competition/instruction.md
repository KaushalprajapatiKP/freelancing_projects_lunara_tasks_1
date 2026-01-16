# Binary Classification with Stroke Prediction Dataset

Predict the probability of stroke occurrence in patients based on various features.

Training data: /app/data/train.csv
Test data: Hidden from you
Output: /app/predict.py that loads your trained model and makes predictions

## Task

1. Train a binary classification model to predict stroke probability
2. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Outputs predictions to /app/predictions.csv
   - Columns: ID,TARGET (TARGET must be probabilities between 0.0 and 1.0)

## Data Format

- Training CSV: ID, feature_1, feature_2, ..., feature_N, stroke (stroke is the target variable, 0 or 1)
- Test CSV: ID, feature_1, feature_2, ..., feature_N (no stroke column)
- Features may include numeric and categorical variables
- Target stroke is binary (0 or 1)

## Evaluation

Your model is evaluated using F1-Score:
- F1-Score balances precision and recall, which is crucial for healthcare applications
- You need F1-Score ≥ 0.305 to pass (binary score: 1.0 if ≥0.305, else 0.0)
- Higher F1-Score is better (maximum is 1.0)
- F1-Score = 2 * (precision * recall) / (precision + recall)
- Note: The dataset is highly imbalanced (~4% positive class), making this a challenging task

## Available packages

Python standard library, NumPy, pandas, scikit-learn, lightgbm, xgboost, joblib

