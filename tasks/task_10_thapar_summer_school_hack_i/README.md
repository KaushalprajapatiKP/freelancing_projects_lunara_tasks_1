# Task: Thapar Summer School Hack-I (Harbor Format)

## Task Format: Harbor

This task follows the **Harbor** task format. See the main Harbor structure in task_0_harbor-predicting-euphoria-in-the-streets README.

## Problem Overview

**Task Type:** Regression  
**Domain:** General Regression / Competition  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Predict the target value based on various features using regression modeling. This task is based on the Thapar Summer School 2025 - Hack-I competition.

## Task Configuration

From `task.toml`:
- **Version:** 1.0
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** regression, machine-learning, competition, tabular-data

## Data Structure

### Training Data
- **Location:** `/app/data/train.csv`
- **Format:** CSV with columns: `id, feature_1, feature_2, ..., feature_N, target`
- **Target Column:** `target` (continuous numeric value)
- **Features:** May include numeric and categorical variables

### Test Data
- **Location:** Hidden from you (grader only)
- **Format:** CSV with columns: `id, feature_1, feature_2, ..., feature_N` (no target column)
- **Features:** Same structure as training data

### Output Format
- **File:** `/app/predict.py` (executable Python script)
- **Output File:** `/app/predictions.csv`
- **Required Columns:**
  - `id`: Sample identifier (must match test data)
  - `target`: Predicted target value (numeric)

Example output:
```csv
id,target
1,1250.5
2,980.3
3,1450.7
```

## Solution Requirements

### 1. Training Script (`solution/solve.sh`)

The solution must:
1. Load training data from `/app/data/train.csv`
2. Handle missing values appropriately
3. Handle categorical variables if present
4. Engineer features appropriately
5. Train a regression model
6. Save the trained model to `/app/model.pkl`
7. Create `/app/predict.py` that:
   - Takes test CSV path as `sys.argv[1]`
   - Loads the trained model
   - Applies the same preprocessing
   - Outputs predictions to `/app/predictions.csv`

### 2. Prediction Script (`/app/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument: `python3 /app/predict.py <test_csv_path>`
- Load model from `/app/model.pkl`
- Load test data from the provided path
- Apply identical preprocessing as training
- Output predictions to `/app/predictions.csv` with format: `id,target`
- Ensure all `target` values are valid numeric values

## Evaluation

### Metric: R² (Coefficient of Determination)

**Formula:**
```
R² = 1 - (SS_res / SS_tot)
```

Where:
- `SS_res = Σ(predicted_i - actual_i)²` (sum of squared residuals)
- `SS_tot = Σ(actual_i - mean(actual))²` (total sum of squares)

### R² Interpretation

- R² ranges from -∞ to 1.0 (higher is better)
- R² = 1.0 means perfect predictions
- R² = 0.0 means model performs as well as predicting the mean
- R² < 0.0 means model performs worse than predicting the mean

### Passing Threshold

- **Passing Requirement:** R² ≥ 0.38

This is a reasonable threshold for regression tasks, indicating that the model explains at least 38% of the variance in the target variable.

### Evaluation Process

The test suite performs:
1. **Existence Check:** Verifies `/app/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Check:** Validates predictions CSV format and column names
4. **Value Validation:** Ensures all predictions are valid numeric values
5. **Completeness Check:** Verifies all test samples have predictions
6. **R² Calculation:** Computes R² and checks threshold

The evaluation also checks:
- All test samples have predictions
- Predictions are valid numeric values
- Output format is correct (id, target)
- No extra predictions for unknown IDs

## Available Packages

The environment includes:
- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- lightgbm
- xgboost
- joblib

**Note:** Deep learning frameworks (TensorFlow, PyTorch) are NOT available.

## Environment Details

### Working Directory
- **Training/Development:** `/app`
- **Test Data:** `/tests`
- **Model Storage:** `/app/model.pkl`
- **Output:** `/app/predictions.csv`

## Technical Considerations

### Data Preprocessing

1. **Missing Values:**
   - Features may contain NaN values
   - Consider median imputation, mean imputation, or advanced techniques
   - Missing values may be informative

2. **Categorical Variables:**
   - Encode categorical variables (one-hot, label encoding, or target encoding)
   - Consider high-cardinality categoricals

3. **Feature Engineering:**
   - Consider log transformations for skewed features
   - Create interaction features
   - Consider polynomial features

4. **Normalization/Scaling:**
   - Scale numeric features if using distance-based models
   - Consider robust scaling for outliers

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Gradient boosting, handles mixed data types well
- **Random Forest:** Robust baseline
- **Gradient Boosting:** Good for non-linear relationships
- **Linear Regression:** Interpretable baseline
- **Ensemble Methods:** Combine multiple models

### Validation Strategy

- Use train/validation split or cross-validation
- Monitor R² on validation set
- Prevent overfitting through regularization
- Consider early stopping for gradient boosting

### Handling Outliers

- Identify and handle outliers in target variable
- Consider robust regression methods
- Use robust scaling for features

## Success Criteria

To pass this task:
1. ✅ `/app/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`id,target`)
4. ✅ All predictions are valid numeric values
5. ✅ All test samples have predictions
6. ✅ **R² ≥ 0.38**

## Competition Information

This task is based on the Thapar Summer School 2025 - Hack-I competition:
- **Problem type:** Regression
- **Evaluation metric:** R² (Coefficient of Determination)
- **Submission format:** CSV with columns `id` and `target`

## Additional Notes

- This is a competition-style regression task
- The threshold of R² ≥ 0.38 indicates the model should explain at least 38% of variance
- Focus on robust preprocessing and model selection
- Consider that target values should be valid numeric values
- The task name suggests this is from a hackathon/competition context


