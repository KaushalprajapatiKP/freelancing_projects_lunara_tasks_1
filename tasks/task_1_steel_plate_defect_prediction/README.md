# Task: Steel Plate Defect Prediction (T-bench Format)

## Task Format: T-bench

This task follows the **T-bench** task format. See the main T-bench structure in task_0_t_bench_taxi_fare_predictor_v1 README.

## Problem Overview

**Task Type:** Multi-label Classification  
**Domain:** Manufacturing / Quality Control  
**Difficulty:** Very Hard  
**Category:** MLE (Machine Learning Engineering)

Build a machine learning system to predict multiple types of defects in steel plates using geometric and visual features. Quality control in steel manufacturing relies on automated defect detection systems to identify various types of faults during production.

### Dataset Citation
This task uses the Steel Plates Faults Data Set from the UCI Machine Learning Repository.
- **Source:** Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults Data Set. UCI Machine Learning Repository. https://doi.org/10.24432/C5389J

## Task Configuration

From `task.yaml` metadata:
- **Difficulty:** very-hard
- **Category:** MLE
- **Tags:** multi-label-classification, machine-learning, defect-detection, manufacturing, imbalanced-data
- **Time Limit:** 300 seconds (5 minutes)
- **Memory Limit:** 512 MB
- **Max Agent Timeout:** 600 seconds (10 minutes)
- **Expert Time Estimate:** 120 minutes
- **Junior Time Estimate:** 240 minutes

## Data Structure

### Training Data
- **Location:** `/workdir/data/train.csv`
- **Samples:** 15,375 steel plates with known defect labels
- **Format:** CSV with features and 7 binary target columns (defect types)

### Test Data
- **Location:** `/tests/test.csv` (hidden during development)
- **Samples:** ~3,844 test plates
- **Format:** CSV with features only (no defect labels)

### Output Format
- **File:** `/workdir/predict.py` (executable Python script)
- **Output File:** `/workdir/predictions.csv`
- **Required Columns:**
  - `id`: Plate identifier
  - `Pastry`: Probability (0.0 to 1.0) for Pastry defect
  - `Z_Scratch`: Probability (0.0 to 1.0) for Z-shaped scratch defect
  - `K_Scratch`: Probability (0.0 to 1.0) for K-shaped scratch defect
  - `Stains`: Probability (0.0 to 1.0) for Stains
  - `Dirtiness`: Probability (0.0 to 1.0) for Dirtiness
  - `Bumps`: Probability (0.0 to 1.0) for Bumps
  - `Other_Faults`: Probability (0.0 to 1.0) for Other types of faults

Example output:
```csv
id,Pastry,Z_Scratch,K_Scratch,Stains,Dirtiness,Bumps,Other_Faults
1,0.12,0.05,0.03,0.89,0.15,0.02,0.01
2,0.01,0.95,0.02,0.10,0.05,0.03,0.12
```

## Features

### Geometric Features (8 columns)
- **X_Minimum, X_Maximum:** X-coordinate bounds of defect
- **Y_Minimum, Y_Maximum:** Y-coordinate bounds of defect
- **Pixels_Areas:** Area of defect in pixels
- **X_Perimeter, Y_Perimeter:** Perimeter measurements
- **Length_of_Conveyer:** Conveyer length measurement

### Visual Features (16 columns)
- **Sum_of_Luminosity, Minimum_of_Luminosity, Maximum_of_Luminosity:** Luminosity statistics
- **Edges_Index, Empty_Index, Square_Index:** Shape indices
- **Outside_X_Index, Edges_X_Index, Edges_Y_Index, Outside_Global_Index:** Edge detection indices
- **LogOfAreas, Log_X_Index, Log_Y_Index:** Logarithmic transformations
- **Orientation_Index, Luminosity_Index, SigmoidOfAreas:** Normalized indices

### Material Properties (3 columns)
- **TypeOfSteel_A300, TypeOfSteel_A400:** Binary indicators for steel type (mutually exclusive)
- **Steel_Plate_Thickness:** Thickness of the steel plate

### Target Variables (7 columns - Multi-label)
- **Pastry:** Binary (0 or 1) - Pastry defect present
- **Z_Scratch:** Binary (0 or 1) - Z-shaped scratch defect
- **K_Scratch:** Binary (0 or 1) - K-shaped scratch defect
- **Stains:** Binary (0 or 1) - Stains present
- **Dirtiness:** Binary (0 or 1) - Dirtiness present
- **Bumps:** Binary (0 or 1) - Bumps present
- **Other_Faults:** Binary (0 or 1) - Other types of faults

**Note:** Multiple defects can be present simultaneously on the same plate (multi-label classification).

## Solution Requirements

### 1. Training Script (`solution.sh`)

The solution must:
1. Load training data from `/workdir/data/train.csv`
2. Handle multi-label classification (7 target columns)
3. Engineer features appropriately
4. Train model(s) for multi-label prediction
5. Save model and preprocessing to `/workdir/model.pkl`
6. Create `/workdir/predict.py` for inference

### 2. Prediction Script (`/workdir/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument
- Load model from `/workdir/model.pkl`
- Output probabilities for all 7 defect types
- Save to `/workdir/predictions.csv` with all 8 columns (id + 7 defect types)

Example structure:
```python
import sys
import pandas as pd
import pickle

model = pickle.load(open('/workdir/model.pkl', 'rb'))
test_df = pd.read_csv(sys.argv[1])

# Make predictions (probabilities for each of 7 defect types)
predictions = model.predict_proba(...)

pd.DataFrame({
    'id': test_df['id'],
    'Pastry': predictions[:, 0],
    'Z_Scratch': predictions[:, 1],
    'K_Scratch': predictions[:, 2],
    'Stains': predictions[:, 3],
    'Dirtiness': predictions[:, 4],
    'Bumps': predictions[:, 5],
    'Other_Faults': predictions[:, 6]
}).to_csv('/workdir/predictions.csv', index=False)
```

## Evaluation

### Primary Metric: Mean ROC-AUC

**Formula:**
- ROC-AUC is calculated separately for each of the 7 defect types
- Then averaged to get the final mean ROC-AUC score
- Higher scores indicate better model performance

### Passing Thresholds

**ALL thresholds must be met:**
1. **Primary:** Mean ROC-AUC ≥ 0.8835 across all defect types
2. **Secondary:** Minimum per-target ROC-AUC ≥ 0.70 for each individual defect type
3. **Fairness:** Maximum variance in per-target ROC-AUC scores ≤ 0.10

### Why These Thresholds Are Challenging

The dataset has significant class imbalance (especially for Other_Faults at ~5% prevalence), requiring:
- Advanced feature engineering (polynomial features, interaction terms, domain-specific ratios)
- Strong ensemble methods (XGBoost, LightGBM with proper hyperparameter tuning)
- Multi-label classification approaches (Classifier Chains for label dependencies, stacking)
- Aggressive class imbalance handling (oversampling with noise augmentation, uncapped class weights)
- Per-target model optimization (different hyperparameters for weak vs strong targets)
- Probability calibration (isotonic regression for better ranking)

The fairness constraint prevents solutions that ignore weak targets to boost the mean score.

## Available Packages

The environment includes:
- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- scipy
- xgboost
- lightgbm

## Environment Details

### Working Directory
- **Training/Development:** `/workdir`
- **Test Data:** `/tests`
- **Model Storage:** `/workdir/model.pkl`
- **Output:** `/workdir/predictions.csv`

## Technical Considerations

### Multi-label Classification Approaches

1. **Binary Relevance:** Train 7 separate binary classifiers
2. **Classifier Chains:** Chain classifiers to capture label dependencies
3. **Multi-output Models:** Use models that natively support multi-label
4. **Ensemble Methods:** Combine multiple approaches

### Handling Class Imbalance

- **Other_Faults** has ~5% prevalence (very imbalanced)
- Use techniques like:
  - Class weights (inverse frequency)
  - SMOTE or other oversampling
  - Focal loss or similar
  - Threshold optimization per target

### Feature Engineering

- Polynomial features and interaction terms
- Domain-specific ratios (e.g., area/perimeter)
- Statistical aggregations
- Normalization and scaling

### Data Quality Notes

- Historical datasets may contain a column named `K_Scatch` (typo). The provided train/test files have been normalized to `K_Scratch`.
- Real-world measurement noise: The dataset contains inherent sensor measurement noise (~5% variation in feature values), reflecting realistic manufacturing conditions. Your model must be robust to this noise.

## Success Criteria

To pass this task:
1. ✅ `/workdir/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (id + 7 defect columns)
4. ✅ All probability values are in [0.0, 1.0]
5. ✅ All test samples have predictions
6. ✅ **Mean ROC-AUC ≥ 0.8835**
7. ✅ **Min per-target ROC-AUC ≥ 0.70**
8. ✅ **Max variance in per-target ROC-AUC ≤ 0.10**

## Additional Notes

- This is a very challenging task due to multi-label classification and class imbalance
- The fairness constraint ensures robust performance across all defect types
- Consider using different models or hyperparameters for different defect types
- Probability calibration may help improve ROC-AUC scores


