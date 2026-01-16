# Task: Predicting Euphoria in the Streets (Harbor Format)

## Task Format: Harbor

This task follows the **Harbor** task format, which uses:
- `task.toml` - Task configuration file
- `instruction.md` - Task instructions
- `environment/Dockerfile` - Container environment definition
- `solution/solve.sh` - Solution script entry point
- `tests/test.sh` - Test execution script
- `tests/test_outputs.py` - Test validation logic

### Harbor Task Structure

```
task_0_harbor-predicting-euphoria-in-the-streets/
├── task.toml                    # Task configuration
├── instruction.md               # Task instructions
├── environment/
│   ├── Dockerfile              # Container environment
│   └── data/
│       └── train.csv           # Training data
├── solution/
│   └── solve.sh                # Solution entry point
└── tests/
    ├── test.sh                 # Test execution script
    ├── test_outputs.py         # Test validation logic
    ├── test.csv                # Test data (without labels)
    └── test_ground_truth.csv   # Ground truth for evaluation
```

## Problem Overview

**Task Type:** Binary Classification  
**Domain:** Healthcare / Tabular Data  
**Difficulty:** Hard  
**Category:** MLE (Machine Learning Engineering)

Predict the probability of euphoria from various features. This is a binary classification problem where the target variable indicates whether euphoria occurs (True/False).

## Task Configuration

From `task.toml`:
- **Version:** 1.0
- **Difficulty:** hard
- **Category:** MLE
- **Tags:** multi-task, regression, classification, healthcare, tabular-data
- **Verifier Timeout:** 3600 seconds (1 hour)
- **Agent Timeout:** 3600 seconds (1 hour)
- **Environment Build Timeout:** 600 seconds (10 minutes)
- **Memory Limit:** 2048 MB

## Data Structure

### Training Data
- **Location:** `/app/data/train.csv`
- **Format:** CSV with columns: `id, x_1, x_2, ..., x_N, Y`
- **Target Column:** `Y` (boolean: True/False)
- **Features:** Numeric features (x_1, x_2, ..., x_N)
- **Data Characteristics:**
  - Features may contain missing values (NaN)
  - Features may contain infinite values (inf, -inf)
  - All features are numeric

### Test Data
- **Location:** Hidden from you (provided by grader at runtime)
- **Format:** CSV with columns: `id, x_1, x_2, ..., x_N` (no Y column)
- **Test data structure matches training data (same feature columns)**

### Output Format
- **File:** `/app/predict.py` (executable Python script)
- **Output File:** `/app/predictions.csv`
- **Required Columns:**
  - `id`: Sample identifier (must match test data)
  - `target`: Probability value between 0.0 and 1.0

Example output:
```csv
id,target
1,0.7234
2,0.1567
3,0.8901
```

## Solution Requirements

### 1. Training Script (`solution/solve.sh`)

The solution must:
1. Load training data from `/app/data/train.csv`
2. Handle missing values appropriately
3. Handle infinite values appropriately
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
- Output predictions to `/app/predictions.csv` with format: `id,target`
- Ensure all `target` values are probabilities between 0.0 and 1.0

Example `predict.py` structure:
```python
import sys
import pandas as pd
import joblib

# Load model
model = joblib.load('/app/model.pkl')

# Load test data
test_df = pd.read_csv(sys.argv[1])

# Preprocess (same as training)
# ... preprocessing steps ...

# Make predictions
predictions = model.predict_proba(test_df)[:, 1]  # Get probability of positive class

# Save predictions
pd.DataFrame({
    'id': test_df['id'],
    'target': predictions
}).to_csv('/app/predictions.csv', index=False)
```

## Evaluation

### Metric: Area Under the ROC Curve (AUC-ROC)

**Formula:**
- AUC-ROC measures the ability to distinguish between positive and negative classes
- Calculated by plotting True Positive Rate (TPR) vs False Positive Rate (FPR) at various thresholds
- Higher AUC-ROC is better (maximum is 1.0)

### Passing Threshold

- **Passing Requirement:** AUC-ROC ≥ 0.787
- **Binary Score:** 
  - 1.0 if AUC-ROC ≥ 0.787
  - 0.0 if AUC-ROC < 0.787

### Evaluation Process

The test suite (`tests/test_outputs.py`) performs:
1. **Existence Check:** Verifies `/app/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Check:** Validates predictions CSV format and column names
4. **Value Range Check:** Ensures all predictions are in [0.0, 1.0]
5. **Completeness Check:** Verifies all test samples have predictions
6. **AUC-ROC Calculation:** Computes AUC-ROC and checks threshold

## Available Packages

The environment includes:
- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- lightgbm
- joblib

**Note:** Deep learning frameworks (TensorFlow, PyTorch) are NOT available.

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

### Data Preprocessing

1. **Missing Values:**
   - Features may contain NaN values
   - Consider median imputation, mean imputation, or advanced techniques
   - Missing values may be informative

2. **Infinite Values:**
   - Features may contain inf or -inf
   - Replace with appropriate values (e.g., large finite numbers, NaN then impute)

3. **Feature Engineering:**
   - Consider log transformations for skewed features
   - Create interaction features
   - Consider polynomial features

### Model Selection

Recommended approaches:
- **LightGBM:** Gradient boosting, handles missing values well
- **XGBoost:** Alternative gradient boosting
- **Ensemble Methods:** Combine multiple models
- **Hyperparameter Tuning:** Use cross-validation

### Validation Strategy

- Use stratified k-fold cross-validation
- Monitor AUC-ROC on validation sets
- Prevent overfitting through regularization

## Test Execution

Tests are run via `tests/test.sh`:
```bash
#!/bin/bash
# Installs dependencies and runs pytest
uvx -p 3.12 -w pytest==8.4.1 -w pandas==2.2.0 pytest /tests/test_outputs.py -rA
```

## Success Criteria

To pass this task:
1. ✅ `/app/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`id,target`)
4. ✅ All predictions are probabilities in [0.0, 1.0]
5. ✅ All test samples have predictions
6. ✅ **AUC-ROC ≥ 0.787**

## Additional Notes

- The task name suggests healthcare/medical context, but the actual features are anonymized (x_1, x_2, etc.)
- Focus on robust preprocessing and model selection rather than domain knowledge
- The threshold of 0.787 is challenging but achievable with proper ML techniques
- Consider class imbalance if present in the data


