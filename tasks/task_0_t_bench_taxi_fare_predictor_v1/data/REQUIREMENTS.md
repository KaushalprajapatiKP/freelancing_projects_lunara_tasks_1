Taxi Fare Prediction - Detailed Requirements

OVERVIEW

Build a machine learning system to predict taxi fare amounts using geospatial, temporal, and contextual features. You'll train a model and create a prediction script (predict.py) that can be executed on hidden test data.

IMPLEMENTATION APPROACH

Step 1: Load Features

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load data
train_df = pd.read_csv('/workdir/data/train_trips.csv')

# Select features (drop IDs and target)
feature_cols = [c for c in train_df.columns if c not in ['trip_id', 'fare_amount']]
X_train = train_df[feature_cols]
y_train = train_df['fare_amount']
```

Step 2: Scale Features

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

Step 3: Train Model

```python
model = LinearRegression()

model.fit(X_train_scaled, y_train)
```

Step 4: Save Model and Create predict.py

```python
# Save model and preprocessing objects
with open('/workdir/model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols
    }, f)

# Create predict.py (see example below)
```

PREDICT.PY TEMPLATE

Your predict.py should follow this structure:

```python
#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import pickle

# Load test data
test_file = sys.argv[1]
test_df = pd.read_csv(test_file)
trip_ids = test_df['trip_id'].values

# Load trained model
with open('/workdir/model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']
    label_encoders = saved_data['label_encoders']
    feature_cols = saved_data['feature_cols']

# Encode categoricals
for col in ['pickup_zone', 'dropoff_zone', 'traffic_level', 'weather_condition', 'trip_type']:
    le = label_encoders[col]
    test_df[col] = test_df[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

# Select and scale features
X_test = test_df[feature_cols]
X_test_scaled = scaler.transform(X_test)

# Predict
predictions = model.predict(X_test_scaled)
predictions = np.maximum(predictions, 1.0)  # Ensure positive

# Save
pd.DataFrame({
    'trip_id': trip_ids,
    'predicted_fare': predictions
}).to_csv('/workdir/predictions.csv', index=False)
```

EVALUATION

Your model is evaluated using Mean Absolute Error (MAE):

MAE Formula: (1/n) * Σ|predicted_i - actual_i|

Scoring:
- MAE ≤ $3.50: 100% score
- MAE ≥ $7.00: 0% score
- Linear scaling between $3.50 and $7

Passing threshold: 96% overall score (MAE ≤ $3.64)

Secondary metrics (informational only):
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination

Why 96% threshold?
- Average fare is ~$35, so $3.64 error is ~10% relative error

OPTIMIZATION TIPS

1. Cross-validation: Use 5-fold CV on training data to validate your approach
