# Task: Country-Level Inflation Prediction (Harbor Format)

## Task Format: Harbor

This task follows the **Harbor** task format. See the main Harbor structure in task_0_harbor-predicting-euphoria-in-the-streets README.

## Problem Overview

**Task Type:** Regression  
**Domain:** Economics / Country Analysis  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Build a machine learning system to predict country-level inflation rates using country metadata and economic indicators. Inflation is a critical economic indicator that reflects the overall increase in prices of goods and services within an economy over a specific period.

## Task Configuration

From `task.toml`:
- **Version:** 1.0
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** regression, machine-learning, economics, tabular-data, country-analysis
- **Verifier Timeout:** 7200 seconds (2 hours)
- **Agent Timeout:** 1200 seconds (20 minutes)
- **Environment Build Timeout:** 600 seconds (10 minutes)
- **Memory Limit:** 1024 MB

## Data Structure

### Training Data
- **Location:** `/app/data/train.csv`
- **Samples:** 118 countries with known inflation rates for 2022
- **Format:** CSV with features and target (`Inflation, 2022` column)

### Test Data
- **Location:** Hidden from you (grader only)
- **Samples:** 30 test countries
- **Format:** CSV with features only (no `Inflation, 2022` column)

### Output Format
- **File:** `/app/predict.py` (executable Python script)
- **Output File:** `/app/predictions.csv`
- **Required Columns:**
  - `Countries`: Country name (must match test file exactly)
  - `Inflation, 2022`: Predicted inflation rate (numeric)

**Important:** The predictions.csv must have the exact header: `Countries,"Inflation, 2022"` (note the comma in the column name and potential quotes)

Example output:
```csv
Countries,"Inflation, 2022"
Sudan,138.8
Zimbabwe,104.7
Turkey,72.3
```

## Features

### Country Features
- **Countries:** Country name (string, categorical)
  - Examples: "Sudan", "Zimbabwe", "Turkey", "USA", "Japan"
  - Requires encoding (e.g., LabelEncoder, OneHotEncoder, or target encoding)
  - May contain special characters and abbreviations

### Economic Indicators
- **Global rank:** Numeric ranking (1-148)
  - Rank 1 = highest inflation rate
  - Rank 148 = lowest inflation rate
  - This is a strong indicator but may not be available for all countries in real scenarios
  - Consider feature engineering: rank categories, rank interactions

### Temporal Features
- **Available data:** String indicating data availability range (e.g., "1960 - 2022", "2010 - 2022")
  - Format: "YYYY - YYYY"
  - Can extract: start year, end year, data span (end - start)
  - Longer data availability may indicate more stable/developed economies
  - Consider parsing and extracting numeric features from this field

### Target Variable
- **Inflation, 2022:** Numeric inflation rate (percentage)
  - Range: approximately 1.0 to 138.8
  - Higher values indicate higher inflation
  - Distribution may be right-skewed

## Feature Engineering Suggestions

### 1. Country Encoding
- **Label encoding** for country names
- **Target encoding** (mean inflation by country) if using cross-validation properly
- **One-hot encoding** if number of countries is manageable
- Consider country groupings (regions, economic blocs) if beneficial

### 2. Available Data Parsing
- Extract `start_year`, `end_year`, `data_span` (end_year - start_year)
- Create categorical features:
  - "long_history" (>=50 years)
  - "medium_history" (20-50 years)
  - "short_history" (<20 years)
- Consider interaction: `rank × data_span`

### 3. Rank Features
- Use rank as-is (numeric)
- Create rank categories:
  - "very_high" (1-30)
  - "high" (31-60)
  - "medium" (61-90)
  - "low" (91-148)
- Consider rank squared or log transformations
- Rank is inversely related to inflation (rank 1 = highest inflation)

### 4. Interaction Features
- `rank × data_span`
- `country_category × rank` (if using country grouping)

## Solution Requirements

### 1. Training Script (`solution/solve.sh`)

The solution must:
1. Load training data from `/app/data/train.csv`
2. Engineer features from all available columns
3. Handle categorical variables (country names) appropriately
4. Parse "Available data" field to extract temporal features
5. Train a regression model
6. Save model and preprocessing to `/app/model.pkl`
7. Create `/app/predict.py` for inference

### 2. Prediction Script (`/app/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument: `python3 /app/predict.py <test_csv_path>`
- Load model from `/app/model.pkl`
- Apply same feature engineering and preprocessing as training
- Output predictions to `/app/predictions.csv`
- **Critical:** Output must have exactly two columns: `Countries, Inflation, 2022`
- Country names must match test file exactly (no duplicates)

Example structure:
```python
import sys
import pandas as pd
import pickle

# Load model and preprocessing objects
with open('/app/model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    # ... load other required preprocessing objects

# Load and validate test data
test_df = pd.read_csv(sys.argv[1])
# Apply feature engineering, handle missing values

predictions = model.predict(...)

pd.DataFrame({
    'Countries': test_df['Countries'],
    'Inflation, 2022': predictions
}).to_csv('/app/predictions.csv', index=False)
```

## Evaluation

### Primary Metric: Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) * Σ|actual_i - predicted_i|
```

**Scoring:**
- MAE ≤ 2.5 → 100% score
- MAE ≥ 8.0 → 0% score
- Linear scaling between 2.5 and 8.0

### Secondary Metrics (Must Also Pass)

**Root Mean Squared Error (RMSE):**
- RMSE = √[(1/n) * Σ(actual - predicted)²]
- **Passing threshold: RMSE ≤ 11.0**

**R² Score (Coefficient of Determination):**
- R² measures how well the model explains variance in the data
- Higher R² is better (maximum is 1.0)
- **Passing threshold: R² ≥ 0.4**

### Passing Requirements

To pass this task, your model must meet **ALL THREE thresholds**:
1. **Primary:** MAE ≤ 6.0
2. **Secondary:** RMSE ≤ 11.0
3. **Secondary:** R² ≥ 0.4

All metrics are calculated and validated during evaluation.

### Evaluation Process

The test suite (`tests/test_outputs.py`) performs:
1. **Existence Check:** Verifies `/app/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Check:** Validates predictions CSV format and column names
4. **Completeness Check:** Verifies all test countries have predictions
5. **MAE Calculation:** Computes MAE and checks threshold (≤ 6.0)
6. **RMSE Calculation:** Computes RMSE and checks threshold (≤ 11.0)
7. **R² Calculation:** Computes R² and checks threshold (≥ 0.4)

## Available Packages

The environment includes:
- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- scipy

**Note:** LightGBM, XGBoost are NOT available in this task.

## Environment Details

### Dockerfile Configuration

The environment is built from `ubuntu:24.04` with:
- Python 3.12
- Build essentials
- Required Python packages (see Available Packages)

### Working Directory
- **Training/Development:** `/app`
- **Test Data:** `/tests`
- **Model Storage:** `/app/model.pkl`
- **Output:** `/app/predictions.csv`

## Technical Considerations

### Model Selection

Recommended approaches:
- **GradientBoostingRegressor:** Good for non-linear relationships
- **RandomForestRegressor:** Robust to overfitting
- **Linear Regression:** Baseline (may not be sufficient)
- **Ensemble Methods:** Combine multiple models

### Handling Right-Skewed Distribution

The target variable (inflation rate) may be right-skewed:
- Consider log transformation of target
- Use robust regression methods
- Consider quantile regression

### Validation Strategy

- Use train/validation split or cross-validation
- Monitor MAE, RMSE, and R² on validation set
- Prevent overfitting through regularization

### Data Quality

- Handle missing values appropriately
- Validate country name matching between train and test
- Ensure no duplicate countries in predictions

## Success Criteria

To pass this task:
1. ✅ `/app/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`Countries, Inflation, 2022`)
4. ✅ All predictions are non-negative numeric values
5. ✅ All test countries have predictions (no duplicates)
6. ✅ Country names match test file exactly
7. ✅ **MAE ≤ 6.0**
8. ✅ **RMSE ≤ 11.0**
9. ✅ **R² ≥ 0.4**

## Dataset Citation

This task uses the Countries Inflation dataset from Hugging Face.
- **Source:** aswin1906/countries-inflation on Hugging Face Datasets
- **License:** Apache 2.0

## Additional Notes

- The task emphasizes feature engineering from limited metadata
- Global rank is a strong feature but may not be available in real scenarios
- Consider that inflation rates can vary dramatically (1.0 to 138.8)
- The three-metric requirement ensures both accuracy (MAE, RMSE) and explanatory power (R²)


