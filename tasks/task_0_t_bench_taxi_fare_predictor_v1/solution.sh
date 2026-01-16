#!/usr/bin/env bash
set -euo pipefail

python3 << 'TRAINING_SCRIPT'
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Loading training data...")

# Load training data
train_df = pd.read_csv('/workdir/data/train_trips.csv')

print(f"Loaded {len(train_df)} training samples")

# Feature engineering function
def engineer_features(df):
    """Extract and engineer features from raw data."""
    df = df.copy()

    # Parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_of_month'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Derived features
    df['speed'] = df['trip_distance_miles'] / (df['trip_duration_minutes'] / 60 + 0.01)
    df['duration_per_mile'] = df['trip_duration_minutes'] / (df['trip_distance_miles'] + 0.01)

    df['surge_multiplier_x_trip_duration_minutes'] = df['surge_multiplier'] * df['trip_duration_minutes']
    df['surge_multiplier_x_trip_distance_miles'] = df['surge_multiplier'] * df['trip_distance_miles']

    return df

print("Engineering features...")
train_df = engineer_features(train_df)

# Encode categorical features
categorical_cols = ['pickup_zone', 'dropoff_zone', 'traffic_level', 'weather_condition', 'trip_type']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    label_encoders[col] = le

# Select features
feature_cols = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
    'passenger_count', 'trip_duration_minutes', 'trip_distance_miles',
    'pickup_zone', 'dropoff_zone', 'traffic_level', 'weather_condition', 'trip_type',
    'surge_multiplier', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'speed', 'duration_per_mile',
    'surge_multiplier_x_trip_duration_minutes', 'surge_multiplier_x_trip_distance_miles'
]

X_train = train_df[feature_cols]
y_train = train_df['fare_amount']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training ensemble models...")

# Train Gradient Boosting model
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    subsample=0.6,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Important features
try:
    print("Important features (in descending order of importance)...")
    gb_importances = gb_model.feature_importances_
    gb_feature_importance_pairs = zip(feature_cols, gb_importances)

    gb_feature_importance_pairs = sorted(gb_feature_importance_pairs, key=lambda x: x[1], reverse=True)

    print("\nGradient Boosting features:")
    for feature, importance in gb_feature_importance_pairs:
        print(f"{feature}: {importance:.6f}")
except Exception as e:
    print(f"Error getting feature importances: {e}")

print("Saving models and encoders...")

# Save ensemble models and preprocessing objects
with open('/workdir/model.pkl', 'wb') as f:
    pickle.dump({
        'gb_model': gb_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols
    }, f)

print("Model trained and saved to /workdir/model.pkl")

TRAINING_SCRIPT

# Create predict.py script
cat > /workdir/predict.py << 'PREDICT_SCRIPT'
#!/usr/bin/env python3
"""
Prediction script for taxi fare prediction.
Usage: python3 predict.py <test_csv_path>
"""

import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

def engineer_features(df):
    """Extract and engineer features from raw data."""
    df = df.copy()

    # Parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_of_month'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Derived features
    df['speed'] = df['trip_distance_miles'] / (df['trip_duration_minutes'] / 60 + 0.01)
    df['duration_per_mile'] = df['trip_duration_minutes'] / (df['trip_distance_miles'] + 0.01)

    df['surge_multiplier_x_trip_duration_minutes'] = df['surge_multiplier'] * df['trip_duration_minutes']
    df['surge_multiplier_x_trip_distance_miles'] = df['surge_multiplier'] * df['trip_distance_miles']

    return df

# Load test data path from command line
if len(sys.argv) < 2:
    print("Usage: python3 predict.py <test_csv_path>")
    sys.exit(1)

test_file = sys.argv[1]

# Load test data
test_df = pd.read_csv(test_file)
trip_ids = test_df['trip_id'].values

# Load trained models
with open('/workdir/model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    gb_model = saved_data['gb_model']
    scaler = saved_data['scaler']
    label_encoders = saved_data['label_encoders']
    feature_cols = saved_data['feature_cols']

# Engineer features
test_df = engineer_features(test_df)

# Encode categorical features
categorical_cols = ['pickup_zone', 'dropoff_zone', 'traffic_level', 'weather_condition', 'trip_type']

for col in categorical_cols:
    le = label_encoders[col]
    test_df[col] = test_df[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

# Select features
X_test = test_df[feature_cols]

# Scale features
X_test_scaled = scaler.transform(X_test)

# Make predictions using ensemble (average of both models)
gb_predictions = gb_model.predict(X_test_scaled)
predictions = gb_predictions

# Ensure positive predictions
predictions = np.maximum(predictions, 1.0)

# Save predictions
output_df = pd.DataFrame({
    'trip_id': trip_ids,
    'predicted_fare': predictions
})

output_df.to_csv('/workdir/predictions.csv', index=False)

print(f"Predictions saved to /workdir/predictions.csv")
PREDICT_SCRIPT

chmod +x /workdir/predict.py

echo "Solution complete! Created:"
echo "  - /workdir/model.pkl (trained model)"
echo "  - /workdir/predict.py (prediction script)"
