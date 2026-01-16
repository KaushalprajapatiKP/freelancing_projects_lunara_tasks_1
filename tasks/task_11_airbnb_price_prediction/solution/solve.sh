#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60, file=sys.stderr)
print("Airbnb Price Prediction - LightGBM Regression", file=sys.stderr)
print("="*60, file=sys.stderr)

print("\nStep 1: Loading training data...", file=sys.stderr)
train_df = pd.read_csv('/app/data/train.csv')
print(f"  Loaded {len(train_df)} training samples", file=sys.stderr)
print(f"  Columns: {list(train_df.columns)}", file=sys.stderr)

print("\nStep 2: Preparing features and target...", file=sys.stderr)
# Identify target column (assuming it's 'price' or similar)
target_col = None
for col in ['price', 'Price', 'PRICE', 'target', 'TARGET', 'y']:
    if col in train_df.columns:
        target_col = col
        break

if target_col is None:
    # Assume last column is target if not found
    target_col = train_df.columns[-1]
    print(f"  Warning: Target column not found, using last column: {target_col}", file=sys.stderr)

feature_cols = [col for col in train_df.columns if col not in [target_col, 'id', 'ID']]
X = train_df[feature_cols].copy()
y = train_df[target_col]

print(f"  Features: {len(feature_cols)}", file=sys.stderr)
print(f"  Target: {target_col}, range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}", file=sys.stderr)

print("\nStep 3: Handling missing values and encoding...", file=sys.stderr)
preprocessing_info = {
    'feature_cols': feature_cols.copy(),
    'numeric_features': [],
    'categorical_features': [],
    'encoders': {},
    'imputers': {},
    'target_col': target_col
}

# Identify numeric and categorical features
for col in feature_cols:
    if X[col].dtype in ['int64', 'float64']:
        preprocessing_info['numeric_features'].append(col)
        # Fill missing values with median
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col].fillna(median_val, inplace=True)
        preprocessing_info['imputers'][col] = median_val
    else:
        preprocessing_info['categorical_features'].append(col)
        # Fill missing values with 'missing'
        X[col].fillna('missing', inplace=True)
        # Label encode categorical features
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        preprocessing_info['encoders'][col] = le

print(f"  Numeric features: {len(preprocessing_info['numeric_features'])}", file=sys.stderr)
print(f"  Categorical features: {len(preprocessing_info['categorical_features'])}", file=sys.stderr)

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)
for col in preprocessing_info['numeric_features']:
    if X[col].isnull().any():
        median_val = preprocessing_info['imputers'][col]
        X[col].fillna(median_val, inplace=True)

print("\nStep 4: Splitting data for validation...", file=sys.stderr)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Training samples: {len(X_train)}", file=sys.stderr)
print(f"  Validation samples: {len(X_val)}", file=sys.stderr)

print("\nStep 5: Training LightGBM model...", file=sys.stderr)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': 42,
    'max_depth': 7
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

print("\nStep 6: Evaluating on validation set...", file=sys.stderr)
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"  Validation RMSE: {val_rmse:.4f}", file=sys.stderr)

print("\nStep 7: Training final model on full data...", file=sys.stderr)
final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    final_train_data,
    num_boost_round=model.best_iteration
)

print("\nStep 8: Saving model and preprocessing info...", file=sys.stderr)
model_data = {
    'model': final_model,
    'preprocessing_info': preprocessing_info
}
joblib.dump(model_data, '/app/model.pkl')
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
        model_data = joblib.load('/app/model.pkl')
        model = model_data['model']
        preprocessing_info = model_data['preprocessing_info']
    except Exception as e:
        print(f"Error loading model artifacts: {e}", file=sys.stderr)
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
        
        # Prepare features
        feature_cols = preprocessing_info['feature_cols']
        X_test = pd.DataFrame(index=test_df.index)
        
        # Process numeric features
        for col in preprocessing_info['numeric_features']:
            if col in test_df.columns:
                X_test[col] = test_df[col].copy()
                # Fill missing values
                median_val = preprocessing_info['imputers'].get(col, 0.0)
                X_test[col].fillna(median_val, inplace=True)
            else:
                # Feature not in test data, use default value
                X_test[col] = preprocessing_info['imputers'].get(col, 0.0)
        
        # Process categorical features
        for col in preprocessing_info['categorical_features']:
            if col in test_df.columns:
                # Fill NaNs and convert to string, handling all edge cases
                X_test[col] = test_df[col].copy()
                # Replace NaN, None, and empty strings with 'missing'
                X_test[col] = X_test[col].fillna('missing')
                X_test[col] = X_test[col].replace(['', 'nan', 'None', None], 'missing')
                X_test[col] = X_test[col].astype(str)
                # Replace any remaining NaN-like strings
                X_test[col] = X_test[col].replace(['nan', 'None', 'NaN', ''], 'missing')
                
                # Use label encoder
                le = preprocessing_info['encoders'][col]
                # Handle unseen categories - map to 'missing' if it exists, otherwise first class
                unique_vals = X_test[col].unique()
                known_vals = set(le.classes_)
                unknown_vals = [v for v in unique_vals if v not in known_vals]
                
                if unknown_vals:
                    # Use 'missing' if it was in training, otherwise use first known class
                    if 'missing' in known_vals:
                        default_val = 'missing'
                    else:
                        default_val = le.classes_[0]
                    X_test[col] = X_test[col].replace(unknown_vals, default_val)
                
                # Transform using label encoder
                X_test[col] = le.transform(X_test[col])
            else:
                # Feature not in test data, use default encoded value
                # Try to use 'missing' encoding if it exists, otherwise 0
                le = preprocessing_info['encoders'][col]
                if 'missing' in le.classes_:
                    X_test[col] = le.transform(['missing'])[0]
                else:
                    X_test[col] = 0
        
        # Handle infinite values
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        for col in preprocessing_info['numeric_features']:
            if X_test[col].isnull().any():
                median_val = preprocessing_info['imputers'].get(col, 0.0)
                X_test[col].fillna(median_val, inplace=True)
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in X_test.columns:
                if col in preprocessing_info['numeric_features']:
                    X_test[col] = preprocessing_info['imputers'].get(col, 0.0)
                else:
                    X_test[col] = 0
        
        # Select features in the correct order
        X_test_final = X_test[feature_cols]
        
        # Make predictions
        predictions = model.predict(X_test_final, num_iteration=model.best_iteration)
        
        # Ensure predictions are non-negative (prices can't be negative)
        predictions = np.maximum(predictions, 0.0)
        
        # Validate predictions - check for any negative values (should be caught by maximum above)
        if np.any(predictions < 0):
            print(f"Warning: Found {np.sum(predictions < 0)} negative predictions, clipping to 0", file=sys.stderr)
            predictions = np.maximum(predictions, 0.0)
        
        # Check prediction statistics
        pred_mean = np.mean(predictions)
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)
        print(f"Prediction statistics: mean={pred_mean:.2f}, min={pred_min:.2f}, max={pred_max:.2f}", file=sys.stderr)
        
        # Ensure we have predictions for all test rows
        if len(predictions) != len(test_df):
            print(f"Warning: Prediction count mismatch. Expected {len(test_df)}, got {len(predictions)}", file=sys.stderr)
            # If predictions are fewer, pad with mean or reasonable default
            if len(predictions) < len(test_df):
                mean_pred = pred_mean if pred_mean > 0 else 100.0
                predictions = np.append(predictions, [mean_pred] * (len(test_df) - len(predictions)))
        
        # Get IDs - ensure they match exactly
        test_ids = test_df[id_col].values
        
        # Create output dataframe with exact ID matching
        output_df = pd.DataFrame({
            'id': test_ids,
            'price': predictions[:len(test_ids)]
        })
        
        # Final validation - ensure all prices are non-negative
        assert np.all(output_df['price'] >= 0), "Found negative prices in output"
        
        # Ensure all IDs are included and no duplicates
        assert len(output_df) == len(test_df), f"Output length {len(output_df)} != test length {len(test_df)}"
        assert len(output_df['id'].unique()) == len(output_df), "Duplicate IDs in output"
        
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
print(f"Validation RMSE: {val_rmse:.4f}", file=sys.stderr)
print("="*60, file=sys.stderr)

PYTHON_EOF

