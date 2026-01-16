# Task: Global EV Charging Stations (T-bench Format)

## Task Format: T-bench

This task follows the **T-bench** task format. See the main T-bench structure in task_0_t_bench_taxi_fare_predictor_v1 README.

## Problem Overview

**Task Type:** Binary Classification  
**Domain:** EV Infrastructure / Sustainability / Geospatial  
**Difficulty:** Hard  
**Category:** Machine Learning

Build a machine learning system to predict the operational status of EV charging stations using structured metadata. As electric vehicle (EV) adoption accelerates worldwide, the reliability of charging infrastructure has become critical for consumers, fleet operators, and urban planners.

## Task Configuration

From `task.yaml` metadata:
- **Difficulty:** hard
- **Category:** machine-learning
- **Tags:** tabular-data, ev-infrastructure, sustainability, classification, classical-ml, geospatial
- **Time Limit:** 7200 seconds (2 hours)
- **Memory Limit:** 2048 MB
- **Max Agent Timeout:** 7200 seconds (2 hours)
- **Expert Time Estimate:** 120 minutes
- **Junior Time Estimate:** 300 minutes

## Problem Statement

Predict the operational status of EV charging stations as a binary classification task:
- **Operational**
- **Not Operational**

Using all available metadata features except the status field itself.

## Data Structure

### Training Data
- **Location:** `/workdir/data/train.csv`
- **Format:** CSV with station metadata and status labels

### Test Data
- **Location:** `/tests/test.csv`
- **Format:** CSV with same schema but without `status` column

### Output Format
- **File:** `/workdir/predict.py` (executable Python script)
- **Output File:** `/workdir/outputs/predictions.csv`
- **Required Columns:**
  - `id`: Station identifier (as string)
  - `prediction`: Either "Operational" or "Not Operational" (exact capitalization)

Example output:
```csv
id,prediction
1,Operational
2,Not Operational
3,Operational
```

## Data Schema

### Required Columns

- **id:** Unique station identifier (string)
- **status:** Operational status (string, ONLY IN TRAINING DATA)
  - Contains many variations: "Operational", "OPERATIONAL", "operational", "Active", "Available", "Not Operational", "Non-Operational", "Out of Service", "Inactive", "Under Maintenance", "Temporarily Closed", "Planned", "Coming Soon", "Under Construction", etc.
  - **CRITICAL:** You must normalize ALL these to "Operational" or "Not Operational"
- **country:** Country name (string)
- **operator:** Charging network operator name (string, may be missing)
- **town:** City/town name (string, may be missing)
- **address:** Street address (string, may be missing)
- **state:** State/province (string, may be missing)
- **postcode:** Postal code (string, may be missing)
- **lat:** Latitude (numeric, may be missing)
- **lon:** Longitude (numeric, may be missing)
- **num_connectors:** Number of charging connectors (numeric, may be missing)
- **connector_types:** Comma or pipe-separated list of connector types (string, may be missing)
  - Common types: "CCS", "Type 2", "CHAdeMO", "Tesla"
- **date_added:** Date when station was added to database (string, ISO format, may be missing)

## Critical Data Quality Issues

### 1. Status Label Variations and Ambiguities

The status field contains many variations beyond basic "Operational" and "Not Operational":
- Functional variants: "Operational", "OPERATIONAL", "operational", "Active", "Available"
- Non-functional variants: "Not Operational", "Non-Operational", "Out of Service", "Inactive"
- Maintenance states: "Under Maintenance", "Temporarily Closed", "Temporarilly Closed" (typo)
- Planning states: "Planned", "Coming Soon", "Under Construction" (should be Not Operational)
- Typos: "Operatinal", "Non-Operatinal", "Temporarilly"
- Mixed case: "OPERATIONAL", "operational", "Operational", "OpErAtIoNaL"

**You must normalize ALL these to "Operational" or "Not Operational"** based on whether the station is currently functional. Stations that are planned, under construction, or under maintenance should be considered "Not Operational".

### 2. Informative Missing Data Patterns

Missing values are NOT random and contain information:
- Missing `operator`: Often indicates less reliable or independent stations (more likely Not Operational)
- Missing `date_added`: May indicate older stations or incomplete records (more likely Not Operational)
- Missing `num_connectors`: Might indicate incomplete setup or unreliable data entry
- Missing `connector_types`: Could indicate stations that are not fully configured

**You should create "is_missing" indicator features** and potentially use different imputation strategies based on what is missing.

### 3. Data Validation Requirements

You MUST validate and handle:
- Coordinate ranges: lat must be between -90 and 90, lon between -180 and 180
- Invalid coordinates: Some entries may have impossible values (lat > 90, lon > 180)
- Negative values: `num_connectors` must be >= 0 (some entries may be negative)
- Future dates: `date_added` should not be in the future (data entry errors)
- Duplicate IDs: Check for and handle duplicate station IDs
- Zero connectors with types: Stations with `num_connectors=0` but `connector_types` listed

### 4. Complex Connector Type Parsing

The `connector_types` field is highly inconsistent and requires robust parsing:
- Mixed separators: "CCS|Type 2,CHAdeMO" (both pipe and comma)
- Abbreviations: "CCS2", "CCS1", "Type2", "Type 2", "Type2 (Mennekes)"
- Duplicates: "CCS, CCS, Type 2" (same connector listed multiple times)
- Special characters: "CCS (Combo)", "Type 2 (Mennekes)", "CHAdeMO v2.0"
- Case variations: "ccs", "CCS", "Ccs", "Type 2", "type 2", "TYPE 2"
- Multiple formats: "CCS|CCS2", "Type2/Type 2", "CHAdeMO,CHAdeMO"

**You must parse these consistently** to extract:
- Individual connector types (CCS, Type 2, CHAdeMO, Tesla)
- Connector count (accounting for duplicates)
- Connector diversity (number of unique types)
- Fast vs slow charging indicators

### 5. Advanced Feature Engineering Requirements

You should create:
- **Interaction features:** `num_connectors * connector_diversity`, `country_operator_interaction`
- **Aggregated features:** Operator reliability (mean operational rate per operator), country reliability (mean operational rate per country), town reliability (with low-count handling)
- **Temporal features:** Days since station was added (handle various date formats), age-based risk indicators
- **Geographic features:** Country-operator interactions, regional patterns
- **Missing data indicators:** Binary features for each column indicating if value is missing

### 6. Threshold Optimization

The default 0.5 probability threshold may NOT be optimal. You must:
- Optimize the classification threshold based on validation performance
- Consider class distribution when selecting threshold
- Use macro F1 score to guide threshold selection
- Potentially use different thresholds for different countries/operators if beneficial

## Solution Requirements

### 1. Training Script (`solution.sh`)

The solution must:
1. Create `/workdir/solution.sh` that implements the full ML pipeline
2. Load and validate the dataset schema
3. Analyze the data (missing values, feature types, class distribution)
4. Preprocess numeric and categorical features appropriately
5. Train one or more classical machine learning models
6. Tune hyperparameters using appropriate methods
7. Select the best-performing model based on cross-validation
8. Save the trained model and preprocessing pipeline to `/workdir/outputs/model.pkl`
9. Create `/workdir/predict.py` that loads the saved model and generates predictions
10. Generate predictions for the test set and save to `/workdir/outputs/predictions.csv`

### 2. Prediction Script (`/workdir/predict.py`)

The prediction script must:
- Usage: `python3 /workdir/predict.py <test_csv_path>`
- Loads the saved model from `/workdir/outputs/model.pkl`
- Applies the same feature engineering and preprocessing as training
- Outputs predictions to `/workdir/outputs/predictions.csv`

## Evaluation

### Metric: Macro-averaged F1 Score

**Formula:**
```
Macro F1 = (F1_Operational + F1_Not_Operational) / 2
```

Where F1 for each class is:
```
F1 = 2 * (precision * recall) / (precision + recall)
```

### Passing Threshold

- **Passing Requirement:** Macro F1 score ≥ 0.5

This threshold ensures overall balanced performance across both classes.

### Why This Threshold Is Challenging

The dataset has approximately 15:1 to 25:1 ratio of operational to non-operational stations. Achieving this threshold requires:
- Sophisticated handling of class imbalance
- Careful feature engineering
- Optimal threshold selection

Simply using default parameters or ignoring data quality issues will NOT suffice.

### Evaluation Process

The evaluation uses a **country-aware split** where entire countries are held out during testing to assess geographic generalization. This means:
- Training and test sets contain different countries
- Your model must generalize across geographic regions
- Country-specific features may not be directly useful

## Available Packages

The environment includes:
- Python 3.x
- NumPy
- pandas
- scikit-learn
- scipy
- joblib
- lightgbm
- xgboost
- imbalanced-learn

**Disallowed:**
- Deep learning frameworks (TensorFlow, PyTorch, JAX)
- External API calls
- Internet access at runtime

## Environment Details

### Working Directory
- **Training/Development:** `/workdir`
- **Test Data:** `/tests`
- **Model Storage:** `/workdir/outputs/model.pkl`
- **Output:** `/workdir/outputs/predictions.csv`

## Technical Considerations

### Handling Class Imbalance

With 15:1 to 25:1 imbalance:
- Use class weights (inverse frequency)
- SMOTE or other oversampling techniques
- Undersampling majority class
- Ensemble methods with balanced sampling
- Focal loss or similar techniques

### Feature Engineering Priorities

1. **Status Normalization:** Critical first step
2. **Connector Type Parsing:** Extract structured features
3. **Missing Data Indicators:** Informative features
4. **Aggregated Features:** Operator/country reliability
5. **Temporal Features:** Age-based indicators
6. **Geographic Features:** Country-operator interactions

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Handle mixed data types well
- **Random Forest:** Robust to missing values
- **Ensemble Methods:** Combine multiple models
- **Threshold Optimization:** Critical for imbalanced data

## Success Criteria

To pass this task:
1. ✅ `/workdir/solution.sh` exists and is executable
2. ✅ `/workdir/predict.py` exists and is executable
3. ✅ Scripts run successfully
4. ✅ Predictions file has correct format (`id,prediction`)
5. ✅ All predictions are "Operational" or "Not Operational"
6. ✅ All test samples have predictions
7. ✅ **Macro F1 score ≥ 0.5**

## Additional Notes

- This task reflects real-world applications in EV infrastructure monitoring, urban mobility analytics, sustainability and energy transition planning, and predictive maintenance systems
- The country-aware split makes this a challenging generalization task
- Data quality issues are intentional and reflect real-world messy data
- The threshold of 0.5 Macro F1 is challenging but achievable with proper techniques


