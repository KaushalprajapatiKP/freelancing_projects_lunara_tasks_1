# NIFTY Stock Movement Prediction from Financial News

Predict stock market movement (Rise, Fall, or Neutral) based on financial news headlines and market context data.

## Overview

The NIFTY dataset contains financial news headlines paired with market context data and stock movement labels. This task requires building a multi-class classification model that predicts whether a stock index will Rise, Fall, or remain Neutral based on news content and historical market indicators.

The dataset combines:
- **News headlines**: Financial news text that may influence market sentiment
- **Market context**: Historical market data including open, high, low, close prices and technical indicators
- **Labels**: Stock movement classification (Rise, Fall, Neutral)

Training data: /app/data/train.csv (1794 samples with labels)
Test data: Hidden from you (317 samples, no labels)
Output: /app/predict.py that loads your trained model and makes predictions

## What you need to do

1. Train a multi-class classification model using:
   - **News text**: Financial news headlines (text feature, may require text processing/encoding)
   - **Context**: Market data in CSV format (date, open, high, low, close, and other indicators)
   - **Date**: Temporal information that may be useful for feature engineering

2. Train a model on the training data and save it to /app/model.pkl
   - The model file must be saved as a pickle file at exactly /app/model.pkl
   - Include all preprocessing objects (text encoders, feature extractors, scalers) needed for prediction
     
3. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Loads the trained model from /app/model.pkl
   - Applies the same feature engineering and preprocessing used during training
   - Outputs predictions to /app/predictions.csv
   - Columns: id,label (exactly two columns)
   - Label values must be one of: "Rise", "Fall", "Neutral"

Example predict.py:
```python
import sys
import pandas as pd
import joblib

test_csv_path = sys.argv[1]

# Load model and preprocessing objects
with open('/app/model.pkl', 'rb') as f:
    model_data = joblib.load(f)
    model = model_data['model']
    # ... load other required preprocessing objects

# Load and validate test data
test_df = pd.read_csv(test_csv_path)
# Apply feature engineering, handle missing values, process text

predictions = model.predict(...)

pd.DataFrame({
    'id': test_df['id'],
    'label': predictions
}).to_csv('/app/predictions.csv', index=False)
```

## Structure of Data

Training Data:
- Location: /app/data/train.csv
- Samples: 1794 samples with known labels
- Format: CSV with columns: id, date, context, news, label, pct_change
- Label distribution: Neutral (977), Rise (457), Fall (360)

Test Data:
- Location: Hidden from you (grader only)
- Samples: 317 test samples
- Format: CSV with columns: id, date, context, news (no label column)

Output:
- File: /app/predict.py (executable Python script)
- Outputs: /app/predictions.csv with exactly two columns: id, label
- The predictions.csv must have the exact header: id,label
- Label values must be exactly: "Rise", "Fall", or "Neutral" (case-sensitive)

## Features

Text Features:
- **news**: Financial news headlines (string, text)
  - May contain multiple sentences or paragraphs
  - Requires text preprocessing (tokenization, vectorization, or embedding)
  - Consider: TF-IDF, word embeddings, or transformer-based features
  - Length varies significantly (369 to 109k characters)

Market Context:
- **context**: Market data in CSV format (string)
  - Contains: date, open, high, low, close prices and technical indicators
  - Format: "date,open,high,low,close,..."
  - Requires parsing to extract numeric features
  - Consider extracting: price changes, volatility, technical indicators

Temporal Features:
- **date**: Date string (format: "YYYY-MM-DD")
  - Can extract: year, month, day, day_of_week
  - May indicate market cycles or seasonal patterns

Target Variable:
- **label**: Multi-class classification target
  - Values: "Rise", "Fall", "Neutral"
  - Class distribution is imbalanced (Neutral is most common)
  - Consider using class weights or stratified sampling

## Feature Engineering Suggestions

1. Text Processing:
   - Extract features from news text using TF-IDF or CountVectorizer
   - Consider sentiment analysis features
   - Extract text length, word count, sentence count
   - Use n-grams for better text representation

2. Context Parsing:
   - Parse the context CSV string to extract numeric features
   - Calculate price changes, returns, volatility
   - Extract technical indicators if present in context

3. Date Features:
   - Extract year, month, day, day_of_week from date
   - Consider cyclical encoding for temporal features

4. Combined Features:
   - Interaction between text sentiment and market context
   - News length vs. market volatility

## Evaluation

Your model is evaluated using **Accuracy**:
- Accuracy = (Number of correct predictions) / (Total number of predictions)
- Higher accuracy is better (maximum is 1.0)
- **Passing threshold: Accuracy ≥ 45.3%**

Scoring uses linear scaling:
- Accuracy ≥ 0.70 gets you 100% score
- Accuracy ≤ 0.40 gets you 0% score
- Linear scaling between 0.40 and 0.70

Note: For a 3-class classification task, 45.3% accuracy is a reasonable threshold. This is better than random guessing of 33.3% and accounts for the challenging nature of financial prediction. The score is calculated as: (accuracy - 0.40) / 0.30, where accuracy ≥ 0.70 gives 100% score and accuracy ≤ 0.40 gives 0% score.

The evaluation also checks:
- All predictions are valid labels ("Rise", "Fall", "Neutral")
- All test samples have predictions
- No extra predictions for unknown IDs

## Available packages

Python standard library, NumPy, pandas, scikit-learn, lightgbm, joblib

## Dataset Citation

This task uses the NIFTY (News-Informed Financial Trend Yield) dataset from Hugging Face.
Source: raeidsaqur/NIFTY on Hugging Face Datasets
Paper: https://arxiv.org/abs/2405.09747
License: MIT

