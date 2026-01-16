#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60, file=sys.stderr)
print("Predicting Euphoria - Improved LightGBM", file=sys.stderr)
print("="*60, file=sys.stderr)

print("\nStep 1: Loading training data...", file=sys.stderr)
train_df = pd.read_csv('/app/data/train.csv')
print(f"  Loaded {len(train_df)} training samples", file=sys.stderr)

print("\nStep 2: Preparing features and target...", file=sys.stderr)
feature_cols = [col for col in train_df.columns if col not in ['id', 'Y']]
X = train_df[feature_cols].copy()
y = train_df['Y'].astype(int)

print("\nStep 3: Creating preprocessing info...", file=sys.stderr)
preprocessing_info = {
    'feature_cols': feature_cols.copy(),
    'feature_medians': {},
    'feature_stats': {}
}

print("\nStep 4: Handling missing values and calculating statistics...", file=sys.stderr)
for col in X.columns:
    preprocessing_info['feature_medians'][col] = X[col].median()
    
    valid_vals = X[col].dropna()
    if len(valid_vals) > 0:
        preprocessing_info['feature_stats'][col] = {
            'mean': valid_vals.mean(),
            'std': valid_vals.std(),
            'min': valid_vals.min(),
            'max': valid_vals.max(),
            'q25': valid_vals.quantile(0.25),
            'q75': valid_vals.quantile(0.75)
        }
    
    if X[col].isnull().any():
        X[col].fillna(preprocessing_info['feature_medians'][col], inplace=True)

print("\nStep 5: Handling infinite values...", file=sys.stderr)
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].isnull().any():
        valid_vals = X[col].dropna()
        if len(valid_vals) > 0:
            upper_bound = valid_vals.quantile(0.99)
            lower_bound = valid_vals.quantile(0.01)
            X.loc[X[col].isnull(), col] = X[col].fillna(upper_bound)
        else:
            X[col].fillna(preprocessing_info['feature_medians'][col], inplace=True)

print("\nStep 6: Creating additional features...", file=sys.stderr)
log_features = []
for col in feature_cols:
    if col in X.columns and (X[col] > 0).all():
        X[f'{col}_log'] = np.log1p(X[col])
        preprocessing_info['feature_cols'].append(f'{col}_log')
        log_features.append(col)
preprocessing_info['log_features'] = log_features

for i in range(1, 6):
    for j in range(i+1, 6):
        col_i = f'x_{i}'
        col_j = f'x_{j}'
        if col_i in X.columns and col_j in X.columns:
            X[f'x_{i}_x_{j}_ratio'] = X[col_i] / (X[col_j] + 1e-10)
            X[f'x_{i}_x_{j}_product'] = X[col_i] * X[col_j]
            if f'x_{i}_x_{j}_ratio' not in preprocessing_info['feature_cols']:
                preprocessing_info['feature_cols'].append(f'x_{i}_x_{j}_ratio')
            if f'x_{i}_x_{j}_product' not in preprocessing_info['feature_cols']:
                preprocessing_info['feature_cols'].append(f'x_{i}_x_{j}_product')

feature_cols = list(X.columns)
preprocessing_info['all_feature_cols'] = feature_cols

print("\nStep 7: Splitting data for validation...", file=sys.stderr)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Training samples: {len(X_train)}", file=sys.stderr)
print(f"  Validation samples: {len(X_val)}", file=sys.stderr)

print("\nStep 8: Training model with early stopping...", file=sys.stderr)
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': 42,
    'max_depth': 6
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

print("\nStep 9: Evaluating on validation set...", file=sys.stderr)
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, y_pred_val)
print(f"  Validation AUC-ROC: {val_auc:.4f}", file=sys.stderr)

print("\nStep 10: Performing 5-fold cross-validation...", file=sys.stderr)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    fold_val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_data)
    
    fold_model = lgb.train(
        lgb_params,
        fold_train_data,
        num_boost_round=model.best_iteration,
        valid_sets=[fold_val_data],
        callbacks=[lgb.log_evaluation(0)]
    )
    
    fold_pred = fold_model.predict(X_fold_val, num_iteration=fold_model.best_iteration)
    fold_auc = roc_auc_score(y_fold_val, fold_pred)
    cv_scores.append(fold_auc)
    print(f"  Fold {fold + 1} AUC-ROC: {fold_auc:.4f}", file=sys.stderr)

print(f"  Mean CV AUC-ROC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})", file=sys.stderr)

print("\nStep 11: Training final model on full data...", file=sys.stderr)
final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    final_train_data,
    num_boost_round=model.best_iteration
)

print("\nStep 12: Saving model and preprocessing info...", file=sys.stderr)
joblib.dump(final_model, '/app/model.pkl')
joblib.dump(preprocessing_info, '/app/preprocessing_info.pkl')
print(f"  ✓ Model saved to /app/model.pkl", file=sys.stderr)
print(f"  ✓ Preprocessing info saved to /app/preprocessing_info.pkl", file=sys.stderr)

print("\nStep 13: Creating predict.py...", file=sys.stderr)
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
        preprocessing_info = joblib.load('/app/preprocessing_info.pkl')
    except Exception as e:
        print(f"Error loading model artifacts: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        test_df = pd.read_csv(test_csv_path)
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        original_feature_cols = preprocessing_info['feature_cols']
        all_feature_cols = preprocessing_info.get('all_feature_cols', original_feature_cols)
        
        X_test = pd.DataFrame(index=test_df.index)
        for col in original_feature_cols:
            if col in test_df.columns:
                X_test[col] = test_df[col].copy()
            else:
                median_val = preprocessing_info['feature_medians'].get(col, 0)
                X_test[col] = median_val
        
        for col in X_test.columns:
            if X_test[col].isnull().any():
                median_val = preprocessing_info['feature_medians'].get(col, 0)
                X_test[col].fillna(median_val, inplace=True)
        
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        for col in X_test.columns:
            if X_test[col].isnull().any():
                valid_vals = X_test[col].dropna()
                if len(valid_vals) > 0:
                    upper_bound = valid_vals.quantile(0.99)
                    X_test.loc[X_test[col].isnull(), col] = X_test[col].fillna(upper_bound)
                else:
                    median_val = preprocessing_info['feature_medians'].get(col, 0)
                    X_test[col].fillna(median_val, inplace=True)
        
        log_features = preprocessing_info.get('log_features', [])
        for base_col in log_features:
            log_col = f'{base_col}_log'
            if base_col in X_test.columns:
                X_test[log_col] = np.log1p(np.maximum(X_test[base_col], 1e-10))
            else:
                median_val = preprocessing_info['feature_medians'].get(base_col, 0)
                X_test[log_col] = np.log1p(max(median_val, 1e-10))
        
        for i in range(1, 6):
            for j in range(i+1, 6):
                col_i = f'x_{i}'
                col_j = f'x_{j}'
                ratio_col = f'x_{i}_x_{j}_ratio'
                product_col = f'x_{i}_x_{j}_product'
                
                if col_i in X_test.columns and col_j in X_test.columns:
                    X_test[ratio_col] = X_test[col_i] / (X_test[col_j] + 1e-10)
                    X_test[product_col] = X_test[col_i] * X_test[col_j]
                else:
                    X_test[ratio_col] = 0.0
                    X_test[product_col] = 0.0
        
        for col in all_feature_cols:
            if col not in X_test.columns:
                X_test[col] = 0.0
        
        X_test_final = X_test[all_feature_cols]
        
        predictions = model.predict(X_test_final, num_iteration=model.best_iteration)
        
        output_df = pd.DataFrame({
            'id': test_df['id'],
            'target': predictions
        })
        
        output_df['target'] = output_df['target'].clip(0.0, 1.0)
        
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
print(f"Validation AUC-ROC: {val_auc:.4f}", file=sys.stderr)
print(f"Mean CV AUC-ROC: {np.mean(cv_scores):.4f}", file=sys.stderr)
print("="*60, file=sys.stderr)

PYTHON_EOF

