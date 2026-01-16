# Task: Taxi Fare Predictor v1 (T-bench Format)

## Task Format: T-bench

This task follows the **T-bench** task format, which uses:
- `task.yaml` - Task configuration and prompt
- `Dockerfile` - Container environment definition
- `grader.py` - Grading/evaluation script
- `solution.sh` - Solution script entry point
- `data/` - Training data directory
- `tests/` - Test data directory

### T-bench Task Structure

```
task_0_t_bench_taxi_fare_predictor_v1/
├── task.yaml                    # Task configuration and prompt
├── Dockerfile                   # Container environment
├── grader.py                    # Evaluation script
├── solution.sh                  # Solution entry point
├── data/
│   ├── train_trips.csv          # Training data
│   └── REQUIREMENTS.md          # Additional requirements
└── tests/
    ├── test_trips.csv           # Test data (without labels)
    └── test_ground_truth.csv    # Ground truth for evaluation
```

## Problem Overview

**Task Type:** Regression  
**Domain:** Transportation / Geospatial / Time Series  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Build a machine learning system to predict taxi fare amounts using geospatial, temporal, and contextual features. Modern taxi and ride-hailing services rely on fare prediction models to support pricing, demand forecasting, and customer transparency.

## Task Configuration

From `task.yaml` metadata:
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** regression, machine-learning, time-series-analysis, data-cleaning
- **Time Limit:** 300 seconds (5 minutes)
- **Memory Limit:** 512 MB
- **Max Agent Timeout:** 600 seconds (10 minutes)
- **Expert Time Estimate:** 90 minutes
- **Junior Time Estimate:** 180 minutes

## Data Structure

### Training Data
- **Location:** `/workdir/data/train_trips.csv`
- **Samples:** 1,500 taxi trips with known fare amounts
- **Format:** CSV with features and target (`fare_amount`)

### Test Data
- **Location:** `/tests/test_trips.csv` (hidden during development)
- **Samples:** ~400 test trips
- **Format:** CSV with features only (no `fare_amount`)

### Output Format
- **File:** `/workdir/predict.py` (executable Python script)
- **Output File:** `/workdir/predictions.csv`
- **Required Columns:**
  - `trip_id`: Trip identifier
  - `predicted_fare`: Predicted fare amount (must be > 0)

Example output:
```csv
trip_id,predicted_fare
1,25.50
2,18.75
3,42.30
```

## Features

### Geospatial Features (8 columns)
- **pickup_latitude:** 40.70 to 40.82 (NYC Manhattan-like coordinates)
- **pickup_longitude:** -74.02 to -73.93
- **dropoff_latitude:** 40.70 to 40.82
- **dropoff_longitude:** -74.02 to -73.93
- **pickup_zone:** Categorical (financial_district, midtown, upper_east, upper_west, downtown, harlem)
- **dropoff_zone:** Same options as pickup_zone
- **trip_distance_miles:** 0.5-25 miles (Haversine distance)

### Temporal Features (1 column)
- **pickup_datetime:** ISO format YYYY-MM-DD HH:MM:SS
  - Training data spans 2024-01-01 to 2024-06-30
  - Extract: hour (0-23), day_of_week (0-6), day_of_month (1-31), month (1-12)
  - Derive: is_weekend, is_rush_hour, is_late_night

### Trip Characteristics (4 columns)
- **passenger_count:** 1-6 passengers
- **trip_duration_minutes:** 2-120 minutes (includes traffic)
- **trip_type:** Categorical (standard, airport, shared, luxury)
- **trip_id:** Unique identifier

### Environmental Context (3 columns)
- **traffic_level:** Categorical (light, moderate, heavy, severe)
- **weather_condition:** Categorical (clear, rain, snow, fog)
- **surge_multiplier:** 1.0-2.5 (dynamic pricing)

### Target Variable
- **fare_amount:** $5-$200 (what you're predicting)

## Solution Requirements

### 1. Training Script (`solution.sh`)

The solution must:
1. Load training data from `/workdir/data/train_trips.csv`
2. Engineer features from raw data:
   - Parse datetime and extract temporal features
   - Encode categorical variables
   - Create interaction features
3. Train a regression model
4. Save model and preprocessing objects to `/workdir/model.pkl`
5. Create `/workdir/predict.py` for inference

### 2. Prediction Script (`/workdir/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument: `python3 /workdir/predict.py <test_csv_path>`
- Load model from `/workdir/model.pkl`
- Apply same feature engineering as training
- Output predictions to `/workdir/predictions.csv`

Example structure:
```python
import sys
import pandas as pd
import pickle

# Load model
with open('/workdir/model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    # ... load preprocessing objects

# Load test data
test_df = pd.read_csv(sys.argv[1])

# Engineer features (same as training)
# ... feature engineering ...

# Make predictions
predictions = model.predict(...)

# Save predictions
pd.DataFrame({
    'trip_id': test_df['trip_id'],
    'predicted_fare': predictions
}).to_csv('/workdir/predictions.csv', index=False)
```

## Feature Engineering Suggestions

### Temporal Features
- Extract hour, day_of_week, day_of_month, month from `pickup_datetime`
- Create binary flags: `is_weekend`, `is_rush_hour`, `is_late_night`
- Cyclic encoding: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### Geospatial Features
- Use coordinates directly or create distance-based features
- Encode zones (one-hot or label encoding)
- Consider zone interactions (pickup_zone × dropoff_zone)

### Interaction Features
- `surge_multiplier × trip_duration_minutes`
- `surge_multiplier × trip_distance_miles`
- `trip_distance_miles / trip_duration_minutes` (speed)
- `trip_duration_minutes / trip_distance_miles` (duration per mile)

### Categorical Encoding
- Label encoding or one-hot encoding for:
  - `pickup_zone`, `dropoff_zone`
  - `traffic_level`, `weather_condition`
  - `trip_type`

## Evaluation

### Primary Metric: Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) * Σ|predicted_fare_i - actual_fare_i|
```

**Scoring:**
- MAE ≤ $3.50 → 100% score
- MAE ≥ $7.00 → 0% score
- Linear scaling between $3.50 and $7.00

### Passing Threshold

- **Passing Requirement:** Score ≥ 96% (equivalent to MAE ≤ $3.64)
- **Binary Score:**
  - 1.0 if score ≥ 96%
  - 0.0 if score < 96%

### Secondary Metrics (Reported but not used for scoring)
- **RMSE (Root Mean Squared Error):** Penalizes large errors more
- **R² (Coefficient of Determination):** Measures variance explained

### Evaluation Process

The grader (`grader.py`) performs:
1. **Existence Check:** Verifies `/workdir/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Validation:** Checks predictions CSV format
4. **Completeness Check:** Verifies all test trips have predictions
5. **Value Validation:** Ensures predicted_fare > 0
6. **Metric Calculation:** Computes MAE, RMSE, R²
7. **Threshold Check:** Applies 96% threshold

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

The environment is built from `apex_arena:base` with:
- Python 3.x
- scikit-learn==1.3.2
- scipy==1.11.4

### Working Directory
- **Training/Development:** `/workdir`
- **Test Data:** `/tests`
- **Model Storage:** `/workdir/model.pkl`
- **Output:** `/workdir/predictions.csv`

## Technical Considerations

### Model Selection

Recommended approaches:
- **GradientBoostingRegressor:** Good for non-linear relationships
- **RandomForestRegressor:** Robust to overfitting
- **Linear Regression:** Baseline (may not be sufficient)
- **Ensemble Methods:** Combine multiple models

### Hyperparameter Tuning

Key hyperparameters to tune:
- Number of estimators
- Learning rate
- Max depth
- Subsample ratio
- Regularization parameters

### Validation Strategy

- Use train/validation split or cross-validation
- Monitor MAE on validation set
- Prevent overfitting through regularization

## Success Criteria

To pass this task:
1. ✅ `/workdir/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`trip_id,predicted_fare`)
4. ✅ All predicted_fare values are positive
5. ✅ All test trips have predictions
6. ✅ **Score ≥ 96% (MAE ≤ $3.64)**

## Additional Notes

- See `/workdir/data/REQUIREMENTS.md` for complete specifications
- The task emphasizes feature engineering from temporal and geospatial data
- Consider that fare is influenced by distance, time, traffic, weather, and surge pricing
- The threshold of 96% score (MAE ≤ $3.64) is challenging but achievable with proper feature engineering


