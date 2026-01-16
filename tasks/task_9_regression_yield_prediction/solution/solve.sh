#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60, file=sys.stderr)
print("Yield Prediction - LightGBM Regression", file=sys.stderr)
print("="*60, file=sys.stderr)

print("\nStep 1: Loading training data...", file=sys.stderr)
train_df = pd.read_csv('/app/data/train.csv')
print(f"  Loaded {len(train_df)} training samples", file=sys.stderr)
print(f"  Columns: {list(train_df.columns)}", file=sys.stderr)

print("\nStep 2: Preparing features and target...", file=sys.stderr)
# Identify target column (assuming it's 'yield' or similar)
target_col = None
for col in ['yield', 'Yield', 'YIELD', 'target', 'TARGET', 'y']:
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
    'scaler': None
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

print("\nStep 5: Training LightGBM model with MAE objective...", file=sys.stderr)
lgb_params = {
    'objective': 'mae',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.015,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': 42
}

num_boost_round = 2500

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=num_boost_round,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
)

print("\nStep 6: Evaluating on validation set...", file=sys.stderr)
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
val_mae = mean_absolute_error(y_val, y_pred_val)
print(f"  Validation MAE: {val_mae:.4f}", file=sys.stderr)

print("\nStep 7: Training final model on full data...", file=sys.stderr)
final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    final_train_data,
    num_boost_round=num_boost_round
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
                X_test[col] = test_df[col].fillna('missing').astype(str)
                # Use label encoder
                le = preprocessing_info['encoders'][col]
                # Handle unseen categories
                unique_vals = set(X_test[col].unique())
                known_vals = set(le.classes_)
                unknown_vals = unique_vals - known_vals
                if unknown_vals:
                    # Map unknown values to a default (use the first known class)
                    X_test[col] = X_test[col].replace(list(unknown_vals), le.classes_[0])
                X_test[col] = le.transform(X_test[col])
            else:
                # Feature not in test data, use default encoded value (0)
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
        predictions = model.predict(X_test_final)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'id': test_df[id_col],
            'yield': predictions
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
print("="*60, file=sys.stderr)

PYTHON_EOF

