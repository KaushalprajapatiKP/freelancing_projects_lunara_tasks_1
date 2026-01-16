# Task: Airbnb Price Prediction (Harbor Format)

## Task Format: Harbor

This task follows the **Harbor** task format. See the main Harbor structure in task_0_harbor-predicting-euphoria-in-the-streets README.

## Problem Overview

**Task Type:** Regression  
**Domain:** Real Estate / Marketplace / Pricing  
**Difficulty:** Medium  
**Category:** MLE (Machine Learning Engineering)

Predict the rental prices for listings on the Airbnb website using various listing features. Accurate price prediction helps both hosts set competitive prices and guests find value, improving the overall marketplace experience.

## Competition Context

This competition is an in-class competition for the course, "Artificial Intelligence and Machine Learning (Spring 2024)", in Renmin University of China.

## Task Configuration

From `task.toml`:
- **Version:** 1.0
- **Difficulty:** medium
- **Category:** MLE
- **Tags:** regression, machine-learning, real-estate, pricing, tabular-data

## Data Structure

### Training Data
- **Location:** `/app/data/train.csv`
- **Format:** CSV with columns: `id, feature_1, feature_2, ..., feature_N, price`
- **Target Column:** `price` (continuous numeric value representing rental price per night)
- **Features:** May include numeric and categorical variables such as:
  - Location (neighborhood, city, coordinates)
  - Property type (apartment, house, room type)
  - Amenities (wifi, kitchen, parking, etc.)
  - Host information (host rating, response rate, etc.)
  - Property characteristics (bedrooms, bathrooms, accommodates, etc.)

### Test Data
- **Location:** Hidden from you (grader only)
- **Format:** CSV with columns: `id, feature_1, feature_2, ..., feature_N` (no price column)
- **Features:** Same structure as training data

### Output Format
- **File:** `/app/predict.py` (executable Python script)
- **Output File:** `/app/predictions.csv`
- **Required Columns:**
  - `id`: Listing identifier (must match test data)
  - `price`: Predicted rental price per night (numeric, should be positive)

Example output:
```csv
id,price
1,125.50
2,89.99
3,250.00
```

## Solution Requirements

### 1. Training Script (`solution/solve.sh`)

The solution must:
1. Load training data from `/app/data/train.csv`
2. Handle missing values appropriately
3. Handle categorical variables (location, property type, amenities, etc.)
4. Engineer features appropriately:
   - Location-based features
   - Property characteristic interactions
   - Amenity combinations
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
- Output predictions to `/app/predictions.csv` with format: `id,price`
- Ensure all `price` values are valid numeric values (typically positive)

Example structure:
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
predictions = model.predict(test_df)

# Ensure positive predictions
predictions = predictions.clip(lower=0)

# Save predictions
pd.DataFrame({
    'id': test_df['id'],
    'price': predictions
}).to_csv('/app/predictions.csv', index=False)
```

## Evaluation

### Metric: Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √[(1/n) * Σ(predicted_i - actual_i)²]
```

### Passing Threshold

- **Passing Requirement:** RMSE ≤ 111.0
- **Lower RMSE is better**

RMSE penalizes large errors more than MAE, making it sensitive to outliers. This is appropriate for pricing where large prediction errors can be costly.

### Evaluation Process

The test suite performs:
1. **Existence Check:** Verifies `/app/predict.py` exists
2. **Execution Check:** Runs `predict.py` with test data
3. **Format Check:** Validates predictions CSV format and column names
4. **Value Validation:** Ensures all predictions are valid numeric values
5. **Completeness Check:** Verifies all test samples have predictions
6. **RMSE Calculation:** Computes RMSE and checks threshold

The evaluation also checks:
- All test samples have predictions
- Predictions are valid numeric values
- Output format is correct (id, price)
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

### Feature Engineering for Real Estate

1. **Location Features:**
   - Encode neighborhood/city (one-hot, label encoding, or target encoding)
   - Use coordinates if available (latitude, longitude)
   - Create location-based aggregations (mean price by neighborhood)

2. **Property Characteristics:**
   - Use bedrooms, bathrooms, accommodates directly
   - Create ratios: price_per_bedroom, price_per_person
   - Create interaction features: bedrooms × bathrooms

3. **Amenities:**
   - Encode amenities (binary features or count)
   - Create amenity combinations
   - Consider premium amenities (pool, gym, etc.)

4. **Host Features:**
   - Use host rating, response rate if available
   - Create host reliability indicators

5. **Temporal Features:**
   - Extract date features if available (season, month, day of week)

### Data Preprocessing

1. **Missing Values:**
   - Handle missing values in amenities, host features
   - Missing values may indicate absence of feature (e.g., no wifi = 0)

2. **Categorical Variables:**
   - High-cardinality: location, property type (use target encoding or frequency encoding)
   - Low-cardinality: room type, property type (use one-hot encoding)

3. **Outliers:**
   - Price outliers may exist (very expensive or very cheap listings)
   - Consider log transformation of target
   - Use robust scaling for features

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Handle mixed data types well, good for tabular data
- **Random Forest:** Robust baseline, handles non-linear relationships
- **Gradient Boosting:** Good for non-linear relationships
- **Ensemble Methods:** Combine multiple models
- **Linear Regression:** Interpretable baseline (may not be sufficient)

### Validation Strategy

- Use train/validation split or cross-validation
- Monitor RMSE on validation set
- Prevent overfitting through regularization
- Consider early stopping for gradient boosting
- Use log-transformed target if prices are right-skewed

### Handling Price Distribution

- Prices are typically right-skewed (few very expensive listings)
- Consider log transformation: `log(price)` as target
- Remember to exponentiate predictions: `exp(predicted_log_price)`
- Ensure predictions are positive

## Success Criteria

To pass this task:
1. ✅ `/app/predict.py` exists and is executable
2. ✅ Script runs successfully with test data
3. ✅ Predictions file has correct format (`id,price`)
4. ✅ All predictions are valid numeric values (typically positive)
5. ✅ All test samples have predictions
6. ✅ **RMSE ≤ 111.0**

## Additional Notes

- This is a real-world pricing prediction task
- The threshold of RMSE ≤ 111.0 depends on the scale of prices (could be in local currency)
- Focus on location-based features as they are strong predictors of price
- Consider that price should be positive (clip negative predictions)
- Property characteristics (bedrooms, bathrooms) are typically strong predictors
- Amenities can add value but may be less predictive than location and size
- This task reflects real-world applications in marketplace pricing, revenue optimization, and competitive analysis
