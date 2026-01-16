# Airbnb Price Prediction

Predict the rental prices for listings on the Airbnb website using various listing features.

## Overview

This competition is an in-class competition for the course, "Artificial Intelligence and Machine Learning (Spring 2024)", in Renmin University of China.

The competition is to predict the rental prices for listings on the Airbnb website. Accurate price prediction helps both hosts set competitive prices and guests find value, improving the overall marketplace experience.

Training data: /app/data/train.csv
Test data: Hidden from you
Output: /app/predict.py that loads your trained model and makes predictions

## Task

1. Train a regression model to predict rental prices for Airbnb listings
2. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Outputs predictions to /app/predictions.csv
   - Columns: id,price (price must be numeric values)
   - Format should match: id,price

## Data Format

- Training CSV: id, feature_1, feature_2, ..., feature_N, price (price is the target variable)
- Test CSV: id, feature_1, feature_2, ..., feature_N (no price column)
- Features may include numeric and categorical variables (e.g., location, property type, amenities, host information)
- Target price is a continuous numeric value representing the rental price per night

## Evaluation

Your model is evaluated using [Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation):
- RMSE = √[(1/n) * Σ(predicted_i - actual_i)²]
- Lower RMSE is better
- **Passing threshold: RMSE ≤ 111.0**

The evaluation checks:
- All test samples have predictions
- Predictions are valid numeric values
- Output format is correct (id, price)
- No extra predictions for unknown IDs

## Available packages

Python standard library, NumPy, pandas, scikit-learn, lightgbm, xgboost, joblib

