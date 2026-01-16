#!/usr/bin/env bash
set -euo pipefail

python3 << 'TRAINING_SCRIPT'
import sys
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

print("Loading training data...")

# Load training data
train_df = pd.read_csv('/app/data/train.csv')

print(f"Loaded {len(train_df)} training samples")
print(f"Columns: {train_df.columns.tolist()}")

# Feature engineering function
def engineer_features(df):
    """Extract and engineer features from raw data."""
    df = df.copy()
    
    # Parse Available data field to extract years
    def parse_data_range(data_str):
        if pd.isna(data_str):
            return None, None, None
        # Pattern: "YYYY - YYYY"
        match = re.search(r'(\d{4})\s*-\s*(\d{4})', str(data_str))
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            span = end_year - start_year
            return start_year, end_year, span
        return None, None, None
    
    # Extract temporal features from Available data
    data_ranges = df['Available data'].apply(parse_data_range)
    df['data_start_year'] = [r[0] if r[0] else 2000 for r in data_ranges]
    df['data_end_year'] = [r[1] if r[1] else 2022 for r in data_ranges]
    df['data_span'] = [r[2] if r[2] else 22 for r in data_ranges]
    
    # Create categorical features for data span
    df['long_history'] = (df['data_span'] >= 50).astype(int)
    df['medium_history'] = ((df['data_span'] >= 20) & (df['data_span'] < 50)).astype(int)
    df['short_history'] = (df['data_span'] < 20).astype(int)
    
    # Rank features
    df['rank_squared'] = df['Global rank'] ** 2
    df['rank_log'] = np.log1p(df['Global rank'])
    
    # Rank categories
    df['rank_very_high'] = (df['Global rank'] <= 30).astype(int)
    df['rank_high'] = ((df['Global rank'] > 30) & (df['Global rank'] <= 60)).astype(int)
    df['rank_medium'] = ((df['Global rank'] > 60) & (df['Global rank'] <= 90)).astype(int)
    df['rank_low'] = (df['Global rank'] > 90).astype(int)
    
    # Interaction features
    df['rank_x_data_span'] = df['Global rank'] * df['data_span']
    df['rank_x_start_year'] = df['Global rank'] * df['data_start_year']
    
    return df

print("Engineering features...")
train_df = engineer_features(train_df)

# Encode country names
print("Encoding categorical features...")
country_encoder = LabelEncoder()
train_df['country_encoded'] = country_encoder.fit_transform(train_df['Countries'].astype(str))

# Select features
feature_cols = [
    'Global rank',
    'data_start_year', 'data_end_year', 'data_span',
    'long_history', 'medium_history', 'short_history',
    'rank_squared', 'rank_log',
    'rank_very_high', 'rank_high', 'rank_medium', 'rank_low',
    'rank_x_data_span', 'rank_x_start_year',
    'country_encoded'
]

X_train = train_df[feature_cols]
y_train = train_df['Inflation, 2022']

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target range: {y_train.min():.2f} to {y_train.max():.2f}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training models...")

# Train Gradient Boosting model
print("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    verbose=1
)
gb_model.fit(X_train_scaled, y_train)

# Train Random Forest model
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate models
print("\nEvaluating models...")
gb_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
rf_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"Gradient Boosting CV MAE: {-gb_scores.mean():.4f} (+/- {gb_scores.std() * 2:.4f})")
print(f"Random Forest CV MAE: {-rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

# Use the better model (lower MAE)
if -gb_scores.mean() <= -rf_scores.mean():
    print("Using Gradient Boosting model")
    best_model = gb_model
else:
    print("Using Random Forest model")
    best_model = rf_model

# Feature importance
try:
    print("\nTop 10 Important features:")
    importances = best_model.feature_importances_
    feature_importance_pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importance_pairs[:10]:
        print(f"  {feature}: {importance:.6f}")
except Exception as e:
    print(f"Error getting feature importances: {e}")

print("\nSaving model and preprocessing objects...")

# Save model and preprocessing objects
with open('/app/model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'country_encoder': country_encoder,
        'feature_cols': feature_cols
    }, f)

print("Model trained and saved to /app/model.pkl")

TRAINING_SCRIPT

# Create predict.py script
cat > /app/predict.py << 'PREDICT_SCRIPT'
#!/usr/bin/env python3
"""
Prediction script for country inflation prediction.
Usage: python3 predict.py <test_csv_path>
"""

import sys
import pandas as pd
import numpy as np
import pickle
import re

def engineer_features(df):
    """Extract and engineer features from raw data."""
    df = df.copy()
    
    # Parse Available data field to extract years
    def parse_data_range(data_str):
        if pd.isna(data_str):
            return None, None, None
        # Pattern: "YYYY - YYYY"
        match = re.search(r'(\d{4})\s*-\s*(\d{4})', str(data_str))
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            span = end_year - start_year
            return start_year, end_year, span
        return None, None, None
    
    # Extract temporal features from Available data
    data_ranges = df['Available data'].apply(parse_data_range)
    df['data_start_year'] = [r[0] if r[0] else 2000 for r in data_ranges]
    df['data_end_year'] = [r[1] if r[1] else 2022 for r in data_ranges]
    df['data_span'] = [r[2] if r[2] else 22 for r in data_ranges]
    
    # Create categorical features for data span
    df['long_history'] = (df['data_span'] >= 50).astype(int)
    df['medium_history'] = ((df['data_span'] >= 20) & (df['data_span'] < 50)).astype(int)
    df['short_history'] = (df['data_span'] < 20).astype(int)
    
    # Rank features
    df['rank_squared'] = df['Global rank'] ** 2
    df['rank_log'] = np.log1p(df['Global rank'])
    
    # Rank categories
    df['rank_very_high'] = (df['Global rank'] <= 30).astype(int)
    df['rank_high'] = ((df['Global rank'] > 30) & (df['Global rank'] <= 60)).astype(int)
    df['rank_medium'] = ((df['Global rank'] > 60) & (df['Global rank'] <= 90)).astype(int)
    df['rank_low'] = (df['Global rank'] > 90).astype(int)
    
    # Interaction features
    df['rank_x_data_span'] = df['Global rank'] * df['data_span']
    df['rank_x_start_year'] = df['Global rank'] * df['data_start_year']
    
    return df

# Load test data path from command line
if len(sys.argv) < 2:
    print("Usage: python3 predict.py <test_csv_path>")
    sys.exit(1)

test_file = sys.argv[1]

# Load test data
test_df = pd.read_csv(test_file)
countries = test_df['Countries'].values

# Load trained model
with open('/app/model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']
    country_encoder = saved_data['country_encoder']
    feature_cols = saved_data['feature_cols']

# Engineer features
test_df = engineer_features(test_df)

# Encode country names (handle unseen countries)
test_df['country_encoded'] = test_df['Countries'].astype(str).apply(
    lambda x: country_encoder.transform([x])[0] if x in country_encoder.classes_ else 0
)

# Select features
X_test = test_df[feature_cols]

# Scale features
X_test_scaled = scaler.transform(X_test)

# Make predictions
predictions = model.predict(X_test_scaled)

# Ensure positive predictions (inflation rates should be positive)
predictions = np.maximum(predictions, 0.1)

# Save predictions
output_df = pd.DataFrame({
    'Countries': countries,
    'Inflation, 2022': predictions
})

output_df.to_csv('/app/predictions.csv', index=False)

print(f"Predictions saved to /app/predictions.csv")
PREDICT_SCRIPT

chmod +x /app/predict.py

echo "Solution complete! Created:"
echo "  - /app/model.pkl (trained model)"
echo "  - /app/predict.py (prediction script)"

