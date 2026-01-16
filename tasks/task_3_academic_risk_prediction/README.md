# Task: Academic Risk Prediction (T-bench Format)

## Task Format: T-bench

This task follows the **T-bench** task format. See the main T-bench structure in task_0_t_bench_taxi_fare_predictor_v1 README.

## Problem Overview

**Task Type:** Classification  
**Domain:** Education / Student Analytics  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Build a machine learning system to predict the academic risk category of students in higher education using demographic, academic, and socioeconomic features. Predicting student academic outcomes is crucial for educational institutions to identify at-risk students early and provide targeted support.

## Task Configuration

From `task.yaml` metadata:
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** classification, education, student-analytics, tabular-data

## Data Structure

### Training Data
- **Location:** `/workdir/data/train.csv`
- **Samples:** 61,214 students with known outcomes
- **Format:** CSV with features and target

### Test Data
- **Location:** `/tests/test.csv`
- **Samples:** ~15,304 students, no labels
- **Format:** CSV with features only

### Output Format
- **File:** `/workdir/predict.py` (executable Python script)
- **Output File:** `/workdir/predictions.csv`
- **Required Columns:** (See task.yaml for exact format)

## Features

### Demographic Features
- **Marital status:** Categorical
- **Gender:** Categorical
- **Age at enrollment:** Numeric
- **Nacionality:** Portuguese spelling, categorical
- **International status:** Binary/categorical

### Academic Features
- **Application mode:** Categorical
- **Course:** Categorical
- **Previous qualification and grade:** Mixed
- **Admission grade:** Numeric

### Family Background
- **Mother's qualification:** Categorical
- **Father's qualification:** Categorical
- **Mother's occupation:** Categorical
- **Father's occupation:** Categorical

### Academic Performance
- **Curricular units (enrolled, evaluations, approved, grades):** Numeric
  - For 1st semester
  - For 2nd semester

### Contextual Features
- **Displaced:** Binary/categorical
- **Educational special needs:** Binary/categorical
- **Debtor:** Binary/categorical
- **Tuition fees status:** Categorical
- **Scholarship holder:** Binary/categorical

### Economic Features
- **Unemployment rate:** Numeric
- **Inflation rate:** Numeric
- **GDP:** Numeric

## Solution Requirements

### 1. Training Script (`solution.sh`)

The solution must:
1. Load training data from `/workdir/data/train.csv`
2. Engineer features from all available columns
3. Handle categorical variables appropriately
4. Train a classification model
5. Save model to `/workdir/model.pkl`
6. Create `/workdir/predict.py` for inference

### 2. Prediction Script (`/workdir/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument
- Load model from `/workdir/model.pkl`
- Apply same preprocessing as training
- Output predictions to `/workdir/predictions.csv`

## Evaluation

See `task.yaml` for detailed evaluation criteria. The model is evaluated using appropriate classification metrics.

## Available Packages

The environment includes:
- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- scipy

## Environment Details

### Working Directory
- **Training/Development:** `/workdir`
- **Test Data:** `/tests`
- **Model Storage:** `/workdir/model.pkl`
- **Output:** `/workdir/predictions.csv`

## Technical Considerations

### Feature Engineering

- Encode categorical variables (one-hot, label encoding, or target encoding)
- Handle missing values appropriately
- Create interaction features if beneficial
- Normalize/scale numeric features

### Model Selection

Recommended approaches:
- **Gradient Boosting:** LightGBM, XGBoost
- **Random Forest:** Robust baseline
- **Logistic Regression:** Interpretable baseline

### Handling Class Imbalance

If the target classes are imbalanced:
- Use class weights
- Apply sampling techniques
- Use appropriate evaluation metrics

## Success Criteria

To pass this task:
1. ✅ `/workdir/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format
4. ✅ All test samples have predictions
5. ✅ Meets evaluation threshold specified in task.yaml

## Additional Notes

- This task focuses on early intervention for at-risk students
- Consider the temporal aspect (1st and 2nd semester performance)
- Economic indicators may provide important context
- Family background features may require careful handling


