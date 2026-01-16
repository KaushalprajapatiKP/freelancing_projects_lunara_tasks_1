# Binary Classification with a Tabular Euphoria Prediction Dataset

Predict the probability of euphoria from various features.

Training data: /app/data/train.csv
Test data: Hidden from you
Output: /app/predict.py that loads your trained model and makes predictions

## Task

1. Train a binary classification model to predict euphoria probability
2. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Outputs predictions to /app/predictions.csv
   - Columns: id,target (target must be probabilities between 0.0 and 1.0)

## Data Format

- Training CSV: id, x_1, x_2, ..., x_N, Y (Y is boolean target)
- Test CSV: id, x_1, x_2, ..., x_N (no Y)
- Features are numeric, may contain missing values and infinite values
- Target Y is boolean (True/False)

## Evaluation

Your model is evaluated using Area Under the ROC Curve (AUC-ROC):
- AUC-ROC measures the ability to distinguish between positive and negative classes
- You need AUC-ROC ≥ 0.787 to pass (binary score: 1.0 if ≥0.787, else 0.0)
- Higher AUC-ROC is better (maximum is 1.0)

## Available packages

Python standard library, NumPy, pandas, scikit-learn, lightgbm, joblib

