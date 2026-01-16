# Country-Level Inflation Prediction

Build a machine learning system to predict country-level inflation rates using country metadata and economic indicators.

## Overview

Inflation is a critical economic indicator that reflects the overall increase in prices of goods and services within an economy over a specific period. Understanding inflation trends on a global scale is crucial for economists, policymakers, investors, and businesses. Accurate inflation prediction enables better economic planning, investment decisions, and policy formulation.

In this challenge, you will build a regression model that predicts inflation rates for countries in 2022 using country metadata and available economic indicators. The model must learn the relationship between country characteristics, global ranking, and historical data availability with inflation rates. The specification below describes the training and test data, required prediction interface, and evaluation criteria.

Training data: /app/data/train.csv (118 countries with known inflation rates)
Test data: Hidden from you (30 countries, no labels)
Output: /app/predict.py that loads your trained model and makes predictions

## What you need to do

1. Train a regression model using country features:
   - Country name: Text feature (may require encoding)
   - Global rank: Numeric ranking based on inflation rate (1 = highest inflation)
   - Available data: Temporal range indicating data availability (e.g., "1960 - 2022")

2. Train a model on the training data and save it to /app/model.pkl
   - The model file must be saved as a pickle file at exactly /app/model.pkl
   - Include all preprocessing objects (encoders, scalers, feature lists) needed for prediction
     
3. Create /app/predict.py that:
   - Takes test CSV path as sys.argv[1]
   - Loads the trained model from /app/model.pkl
   - Applies the same feature engineering and preprocessing used during training
   - Outputs predictions to /app/predictions.csv
   - Columns: Countries, Inflation, 2022 (exactly two columns)
   - Each country must be unique (no duplicates)
   - The Countries values must match the test file exactly

Example predict.py:
```python
import sys, pandas as pd, pickle

test_csv_path = sys.argv[1]

# Load model and preprocessing objects
with open('/app/model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    # ... load other required preprocessing objects

# Load and validate test data
test_df = pd.read_csv(test_csv_path)
# Apply feature engineering, handle missing values

predictions = model.predict(...)

pd.DataFrame({
    'Countries': test_df['Countries'],
    'Inflation, 2022': predictions
}).to_csv('/app/predictions.csv', index=False)
```

## Structure of Data

Training Data:
- Location: /app/data/train.csv
- Samples: 118 countries with known inflation rates for 2022
- Format: CSV with features and target (Inflation, 2022 column)

Test Data:
- Location: Hidden from you (grader only)
- Samples: 30 test countries
- Format: CSV with features only (no Inflation, 2022 column)

Output:
- File: /app/predict.py (executable Python script)
- Outputs: /app/predictions.csv with exactly two columns: Countries, Inflation, 2022
- The predictions.csv must have the exact header: Countries,"Inflation, 2022"
- Each row must have a unique country value (no duplicate countries)
- The Countries values must match the test file exactly

## Features

Country Features:
- Countries: Country name (string, categorical)
  Examples: "Sudan", "Zimbabwe", "Turkey", "USA", "Japan"
  - Requires encoding (e.g., LabelEncoder, OneHotEncoder, or target encoding)
  - May contain special characters and abbreviations

Economic Indicators:
- Global rank: Numeric ranking (1-148)
  - Rank 1 = highest inflation rate
  - Rank 148 = lowest inflation rate
  - This is a strong indicator but may not be available for all countries in real scenarios
  - Consider feature engineering: rank categories, rank interactions

Temporal Features:
- Available data: String indicating data availability range (e.g., "1960 - 2022", "2010 - 2022")
  - Format: "YYYY - YYYY"
  - Can extract: start year, end year, data span (end - start)
  - Longer data availability may indicate more stable/developed economies
  - Consider parsing and extracting numeric features from this field

Target Variable:
- Inflation, 2022: Numeric inflation rate (percentage)
  - Range: approximately 1.0 to 138.8
  - Higher values indicate higher inflation
  - Distribution may be right-skewed

## Feature Engineering Suggestions

1. Country Encoding:
   - Label encoding for country names
   - Consider target encoding (mean inflation by country) if using cross-validation properly
   - One-hot encoding if number of countries is manageable

2. Available Data Parsing:
   - Extract start_year, end_year, data_span
   - Create categorical features: "long_history" (>=50 years), "medium_history" (20-50 years), "short_history" (<20 years)
   - Consider interaction: rank * data_span

3. Rank Features:
   - Use rank as-is (numeric)
   - Create rank categories: "very_high" (1-30), "high" (31-60), "medium" (61-90), "low" (91-148)
   - Consider rank squared or log transformations

4. Interaction Features:
   - rank * data_span
   - country_category * rank (if using country grouping)

## Evaluation

Your model is evaluated using multiple metrics:

### Primary Metric - Mean Absolute Error (MAE):
- MAE = (1/n) * Σ|actual - predicted|
- Lower MAE is better
- **Passing threshold: MAE ≤ 6.0**

Scoring uses linear scaling:
- MAE ≤ 2.5 gets you 100% score
- MAE ≥ 8.0 gets you 0% score
- Linear scaling between 2.5 and 8.0

### Secondary Metrics (must also pass):

**Root Mean Squared Error (RMSE):**
- RMSE = √[(1/n) * Σ(actual - predicted)²]
- Lower RMSE is better
- **Passing threshold: RMSE ≤ 11.0**

**R² Score (Coefficient of Determination):**
- R² measures how well the model explains variance in the data
- Higher R² is better (maximum is 1.0)
- **Passing threshold: R² ≥ 0.4**

### Passing Requirements:
To pass this task, your model must meet **ALL THREE thresholds**:
1. Primary: MAE ≤ 6.0
2. Secondary: RMSE ≤ 11.0
3. Secondary: R² ≥ 0.4

All metrics are calculated and validated during evaluation.

## Available packages

Python standard library, NumPy, pandas, scikit-learn, scipy

## Dataset Citation

This task uses the Countries Inflation dataset from Hugging Face.
Source: aswin1906/countries-inflation on Hugging Face Datasets
License: Apache 2.0

