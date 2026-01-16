#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60, file=sys.stderr)
print("Stock Price Prediction - GradientBoosting Regression", file=sys.stderr)
print("="*60, file=sys.stderr)

print("\nStep 1: Loading training data...", file=sys.stderr)
train_df = pd.read_csv('/app/data/train.csv')
print(f"  Loaded {len(train_df)} training samples", file=sys.stderr)
print(f"  Columns: {list(train_df.columns)}", file=sys.stderr)

print("\nStep 2: Preparing features and target...", file=sys.stderr)
# Identify target column
target_col = None
for col in ['target', 'TARGET', 'Target', 'y']:
    if col in train_df.columns:
        target_col = col
        break

if target_col is None:
    # Assume last column is target if not found
    target_col = train_df.columns[-1]
    print(f"  Warning: Target column not found, using last column: {target_col}", file=sys.stderr)

feature_cols = [col for col in train_df.columns if col not in [target_col, 'id', 'ID', 'Id']]
X = train_df[feature_cols].copy()
y = train_df[target_col]

print(f"  Features: {len(feature_cols)}", file=sys.stderr)
print(f"  Target: {target_col}, range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}", file=sys.stderr)

print("\nStep 3: Handling missing values...", file=sys.stderr)
# Fill missing values with median for numeric features
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col].fillna(median_val, inplace=True)

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col].fillna(median_val, inplace=True)

print("\nStep 4: Splitting data for validation...", file=sys.stderr)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Training samples: {len(X_train)}", file=sys.stderr)
print(f"  Validation samples: {len(X_val)}", file=sys.stderr)

print("\nStep 5: Training GradientBoosting model...", file=sys.stderr)
# Use parameters that work well for this dataset
model = GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

print("\nStep 6: Evaluating on validation set...", file=sys.stderr)
y_pred_val = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
val_r2 = r2_score(y_val, y_pred_val)
print(f"  Validation MAE: {val_mae:.4f}", file=sys.stderr)
print(f"  Validation R²: {val_r2:.4f}", file=sys.stderr)

print("\nStep 7: Training final model on full data...", file=sys.stderr)
final_model = GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
final_model.fit(X, y)

print("\nStep 8: Saving model...", file=sys.stderr)
joblib.dump(final_model, '/app/model.pkl')
print(f"  ✓ Model saved to /app/model.pkl", file=sys.stderr)

print("\nStep 9: Creating predict.py...", file=sys.stderr)
predict_py_content = '''import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <test_csv_path>", file=sys.stderr)
        sys.exit(1)
    
    test_csv_path = sys.argv[1]
    
    try:
        model = joblib.load('/app/model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        test_df = pd.read_csv(test_csv_path)
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Get id column (handle different case variations)
        id_col = None
        for col in ['id', 'ID', 'Id']:
            if col in test_df.columns:
                id_col = col
                break
        
        if id_col is None:
            raise ValueError("Could not find id column in test data")
        
        # Prepare features: drop id and target (if present)
        cols_to_drop = [id_col]
        if 'target' in test_df.columns:
            cols_to_drop.append('target')
        if 'TARGET' in test_df.columns:
            cols_to_drop.append('TARGET')
        if 'Target' in test_df.columns:
            cols_to_drop.append('Target')
        
        X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
        
        # Handle missing values (use median from training - here we use a simple approach)
        for col in X_test.columns:
            if X_test[col].dtype in ['int64', 'float64']:
                median_val = X_test[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_test[col].fillna(median_val, inplace=True)
        
        # Handle infinite values
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        for col in X_test.columns:
            if X_test[col].isnull().any():
                median_val = X_test[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_test[col].fillna(median_val, inplace=True)
        
        # Ensure columns are in the same order as training
        # This assumes the columns match - if not, we'd need to save feature names
        # For simplicity, we assume test data has same features as training
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'id': test_df[id_col],
            'target': predictions
        })
        
        output_df.to_csv('/app/predictions.csv', index=False)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

with open('/app/predict.py', 'w') as f:
    f.write(predict_py_content)

import os
os.chmod('/app/predict.py', 0o755)
print(f"  ✓ Created /app/predict.py", file=sys.stderr)

print("\n" + "="*60, file=sys.stderr)
print("Solution completed successfully!", file=sys.stderr)
print(f"Validation MAE: {val_mae:.4f}", file=sys.stderr)
print(f"Validation R²: {val_r2:.4f}", file=sys.stderr)
print("="*60, file=sys.stderr)

PYTHON_EOF
