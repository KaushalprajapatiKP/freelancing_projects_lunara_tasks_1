# Task: NIFTY Stock Movement Prediction (Harbor Format)

## Task Format: Harbor

This task follows the **Harbor** task format. See the main Harbor structure in task_0_harbor-predicting-euphoria-in-the-streets README.

## Problem Overview

**Task Type:** Multi-class Classification  
**Domain:** Finance / Stock Market / Natural Language Processing  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Predict stock market movement (Rise, Fall, or Neutral) based on financial news headlines and market context data. The NIFTY dataset contains financial news headlines paired with market context data and stock movement labels.

## Task Configuration

From `task.toml`:
- **Version:** 1.0
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** multi-class-classification, finance, nlp, text-classification, stock-market

## Data Structure

### Training Data
- **Location:** `/app/data/train.csv`
- **Samples:** 1,794 samples with labels
- **Format:** CSV with columns: `id, date, context, news, label, pct_change`
- **Label Distribution:**
  - Neutral: 977 samples
  - Rise: 457 samples
  - Fall: 360 samples

### Test Data
- **Location:** Hidden from you (grader only)
- **Samples:** 317 test samples
- **Format:** CSV with columns: `id, date, context, news` (no label column)

### Output Format
- **File:** `/app/predict.py` (executable Python script)
- **Output File:** `/app/predictions.csv`
- **Required Columns:**
  - `id`: Sample identifier
  - `label`: One of "Rise", "Fall", or "Neutral" (case-sensitive)

Example output:
```csv
id,label
1,Rise
2,Neutral
3,Fall
```

## Features

### Text Features
- **news:** Financial news headlines (string, text)
  - May contain multiple sentences or paragraphs
  - Requires text preprocessing (tokenization, vectorization, or embedding)
  - Consider: TF-IDF, word embeddings, or transformer-based features
  - Length varies significantly (369 to 109k characters)

### Market Context
- **context:** Market data in CSV format (string)
  - Contains: date, open, high, low, close prices and technical indicators
  - Format: "date,open,high,low,close,..."
  - Requires parsing to extract numeric features
  - Consider extracting: price changes, volatility, technical indicators

### Temporal Features
- **date:** Date string (format: "YYYY-MM-DD")
  - Can extract: year, month, day, day_of_week
  - May indicate market cycles or seasonal patterns

### Target Variable
- **label:** Multi-class classification target
  - Values: "Rise", "Fall", "Neutral"
  - Class distribution is imbalanced (Neutral is most common)
  - Consider using class weights or stratified sampling

## Feature Engineering Suggestions

### 1. Text Processing
- Extract features from news text using TF-IDF or CountVectorizer
- Consider sentiment analysis features
- Extract text length, word count, sentence count
- Use n-grams for better text representation
- Consider text cleaning (lowercase, remove special characters)

### 2. Context Parsing
- Parse the context CSV string to extract numeric features
- Calculate price changes, returns, volatility
- Extract technical indicators if present in context
- Handle missing or malformed context data

### 3. Date Features
- Extract year, month, day, day_of_week from date
- Consider cyclical encoding for temporal features
- Create time period features (quarter, season)

### 4. Combined Features
- Interaction between text sentiment and market context
- News length vs. market volatility
- Date-based patterns (e.g., end of month, earnings season)

## Solution Requirements

### 1. Training Script (`solution/solve.sh`)

The solution must:
1. Load training data from `/app/data/train.csv`
2. Process text features (news column)
3. Parse context CSV string to extract market features
4. Engineer temporal features from date
5. Combine all features appropriately
6. Train a multi-class classification model
7. Save model and preprocessing to `/app/model.pkl`
8. Create `/app/predict.py` for inference

### 2. Prediction Script (`/app/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument: `python3 /app/predict.py <test_csv_path>`
- Load model from `/app/model.pkl`
- Apply same text processing, context parsing, and feature engineering as training
- Output predictions to `/app/predictions.csv`
- Ensure label values are exactly "Rise", "Fall", or "Neutral" (case-sensitive)

Example structure:
```python
import sys
import pandas as pd
import joblib

# Load model and preprocessing objects
with open('/app/model.pkl', 'rb') as f:
    model_data = joblib.load(f)
    model = model_data['model']
    # ... load text vectorizer, context parser, etc.

# Load and validate test data
test_df = pd.read_csv(sys.argv[1])

# Process text features
# Parse context
# Engineer features

predictions = model.predict(...)

pd.DataFrame({
    'id': test_df['id'],
    'label': predictions
}).to_csv('/app/predictions.csv', index=False)
```

## Evaluation

### Metric: Accuracy

**Formula:**
```
Accuracy = (Number of correct predictions) / (Total number of predictions)
```

### Passing Threshold

- **Passing Requirement:** Accuracy ≥ 45.3%

### Scoring

Scoring uses linear scaling:
- Accuracy ≥ 0.70 → 100% score
- Accuracy ≤ 0.40 → 0% score
- Linear scaling between 0.40 and 0.70

### Why This Threshold

For a 3-class classification task, 45.3% accuracy is a reasonable threshold. This is better than random guessing of 33.3% and accounts for the challenging nature of financial prediction.

The score is calculated as: `(accuracy - 0.40) / 0.30`, where accuracy ≥ 0.70 gives 100% score and accuracy ≤ 0.40 gives 0% score.

### Evaluation Process

The test suite performs:
1. **Existence Check:** Verifies `/app/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Check:** Validates predictions CSV format and column names
4. **Label Validation:** Ensures all predictions are valid labels ("Rise", "Fall", "Neutral")
5. **Completeness Check:** Verifies all test samples have predictions
6. **Accuracy Calculation:** Computes accuracy and checks threshold

## Available Packages

The environment includes:
- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- lightgbm
- joblib

**Note:** Deep learning frameworks (TensorFlow, PyTorch) and advanced NLP libraries may NOT be available. Use scikit-learn's text processing tools (TF-IDF, CountVectorizer).

## Environment Details

### Working Directory
- **Training/Development:** `/app`
- **Test Data:** `/tests`
- **Model Storage:** `/app/model.pkl`
- **Output:** `/app/predictions.csv`

## Technical Considerations

### Text Processing

With limited NLP libraries:
- Use scikit-learn's `TfidfVectorizer` or `CountVectorizer`
- Consider n-grams (unigrams, bigrams, trigrams)
- Limit vocabulary size to manage memory
- Consider text length and word count as features

### Handling Class Imbalance

With imbalanced classes (Neutral: 977, Rise: 457, Fall: 360):
- Use class weights (inverse frequency)
- Apply stratified sampling
- Use appropriate evaluation metrics

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Can handle mixed feature types (text vectors + numeric)
- **Random Forest:** Robust baseline
- **Logistic Regression:** Interpretable baseline
- **Ensemble Methods:** Combine multiple models

### Context Parsing

- Parse CSV string to extract numeric features
- Handle missing or malformed context data
- Calculate derived features (returns, volatility)

## Success Criteria

To pass this task:
1. ✅ `/app/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`id,label`)
4. ✅ All label values are "Rise", "Fall", or "Neutral" (case-sensitive)
5. ✅ All test samples have predictions
6. ✅ **Accuracy ≥ 45.3%**

## Dataset Citation

This task uses the NIFTY (News-Informed Financial Trend Yield) dataset from Hugging Face.
- **Source:** raeidsaqur/NIFTY on Hugging Face Datasets
- **Paper:** https://arxiv.org/abs/2405.09747
- **License:** MIT

## Additional Notes

- This task combines text classification with numeric market data
- Financial prediction is inherently challenging due to market volatility
- The threshold of 45.3% accuracy accounts for the difficulty of predicting market movements
- Consider that news sentiment and market context both contribute to stock movement
- The imbalanced class distribution (Neutral is most common) should be handled appropriately


