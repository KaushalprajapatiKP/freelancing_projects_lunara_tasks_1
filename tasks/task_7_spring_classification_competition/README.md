# Task: Spring Classification Competition (Harbor Format)

## Task Format: Harbor

This task follows the **Harbor** task format. See the main Harbor structure in task_0_harbor-predicting-euphoria-in-the-streets README.

## Problem Overview

**Task Type:** Binary Classification  
**Domain:** Healthcare / Medical  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Predict the probability of stroke occurrence in patients based on various features. This is a binary classification problem where the target variable indicates whether a stroke occurs (0 or 1).

## Task Configuration

From `task.toml`:
- **Version:** 1.0
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** binary-classification, healthcare, medical, tabular-data

## Data Structure

### Training Data
- **Location:** `/app/data/train.csv`
- **Format:** CSV with columns: `ID, feature_1, feature_2, ..., feature_N, stroke`
- **Target Column:** `stroke` (binary: 0 or 1)

### Test Data
- **Location:** Hidden from you (grader only)
- **Format:** CSV with columns: `ID, feature_1, feature_2, ..., feature_N` (no stroke column)
- **Features:** May include numeric and categorical variables

### Output Format
- **File:** `/app/predict.py` (executable Python script)
- **Output File:** `/app/predictions.csv`
- **Required Columns:**
  - `ID`: Sample identifier (must match test data)
  - `TARGET`: Probability value between 0.0 and 1.0

Example output:
```csv
ID,TARGET
1,0.7234
2,0.1567
3,0.8901
```

## Solution Requirements

### 1. Training Script (`solution/solve.sh`)

The solution must:
1. Load training data from `/app/data/train.csv`
2. Handle missing values appropriately
3. Handle categorical variables if present
4. Train a binary classification model
5. Save the trained model to `/app/model.pkl`
6. Create `/app/predict.py` that:
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
- Output predictions to `/app/predictions.csv` with format: `ID,TARGET`
- Ensure all `TARGET` values are probabilities between 0.0 and 1.0

## Evaluation

### Metric: F1-Score

**Formula:**
```
F1 = 2 * (precision * recall) / (precision + recall)
```

Where:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

### Passing Threshold

- **Passing Requirement:** F1-Score ≥ 0.305
- **Binary Score:**
  - 1.0 if F1-Score ≥ 0.305
  - 0.0 if F1-Score < 0.305

### Why This Threshold

- The dataset is highly imbalanced (~4% positive class)
- F1-Score balances precision and recall, which is crucial for healthcare applications
- The threshold of 0.305 accounts for the challenging nature of imbalanced classification

### Evaluation Process

The test suite performs:
1. **Existence Check:** Verifies `/app/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Check:** Validates predictions CSV format and column names
4. **Value Range Check:** Ensures all predictions are in [0.0, 1.0]
5. **Completeness Check:** Verifies all test samples have predictions
6. **F1-Score Calculation:** Computes F1-Score and checks threshold

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

### Handling Class Imbalance

With ~4% positive class:
- Use class weights (inverse frequency)
- Apply SMOTE or other oversampling techniques
- Use appropriate evaluation metrics (F1-Score)
- Consider threshold optimization (not necessarily 0.5)

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Handle mixed data types well, support class weights
- **Random Forest:** Robust baseline
- **Logistic Regression:** Interpretable baseline
- **Ensemble Methods:** Combine multiple models

### Feature Engineering

- Handle missing values appropriately
- Encode categorical variables (one-hot, label encoding, or target encoding)
- Consider feature interactions
- Normalize/scale numeric features

### Validation Strategy

- Use stratified k-fold cross-validation
- Monitor F1-Score on validation sets
- Prevent overfitting through regularization
- Optimize classification threshold based on F1-Score

## Success Criteria

To pass this task:
1. ✅ `/app/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`ID,TARGET`)
4. ✅ All predictions are probabilities in [0.0, 1.0]
5. ✅ All test samples have predictions
6. ✅ **F1-Score ≥ 0.305**

## Additional Notes

- This is a healthcare application where false positives and false negatives both matter
- The imbalanced nature (~4% positive) makes this challenging
- Focus on proper handling of class imbalance
- Consider that stroke prediction is a critical medical application


