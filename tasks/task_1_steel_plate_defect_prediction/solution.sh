#!/usr/bin/env bash
set -euo pipefail

python3 << 'TRAINING_SCRIPT'
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('/workdir/data/train.csv')
if 'K_Scatch' in df.columns and 'K_Scratch' not in df.columns:
    df = df.rename(columns={'K_Scatch': 'K_Scratch'})

targets = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

def create_features(df):
    """Feature engineering with polynomial features"""
    df = df.copy()
    # Basic geometric features
    df['X_Range'] = df['X_Maximum'] - df['X_Minimum']
    df['Y_Range'] = df['Y_Maximum'] - df['Y_Minimum']
    df['Area_Perimeter_Ratio'] = df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'] + 1e-5)
    df['Luminosity_Range'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']
    df['Density'] = df['Pixels_Areas'] / (df['X_Range'] * df['Y_Range'] + 1e-5)
    df['Avg_Luminosity'] = df['Sum_of_Luminosity'] / (df['Pixels_Areas'] + 1e-5)
    df['Orientation_Luminosity'] = df['Orientation_Index'] * df['Luminosity_Index']
    df['Log_Area_Perimeter'] = np.log1p(df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'] + 1e-5))
    df['X_Center'] = (df['X_Maximum'] + df['X_Minimum']) / 2.0
    df['Y_Center'] = (df['Y_Maximum'] + df['Y_Minimum']) / 2.0
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Polynomial features for geometric columns (degree 2, interaction only)
    poly_cols = ['X_Range', 'Y_Range', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[poly_cols])
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    
    # Keep only interaction terms (those with ' ')
    new_cols = [c for c in poly_feature_names if ' ' in c]
    for i, col in enumerate(new_cols):
        df[col] = poly_features[:, poly_feature_names == col][:, 0]
    
    # Replace any new infinite values
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

print("Feature engineering...")
df = create_features(df)

# Get feature columns (exclude targets and id)
features = [col for col in df.columns if col not in targets and col != 'id']
X = df[features].values
y = df[targets]

print(f"Total features: {len(features)}")

# Create models directory
if not os.path.exists('/workdir/models'):
    os.makedirs('/workdir/models')

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store models
final_models = {}
all_aucs = {}

print("\nTraining models with 5-fold CV...")

for target in targets:
    print(f"Training for {target}...")
    y_target = y[target].values
    target_models = {'lgb': [], 'xgb': []}
    fold_aucs = {'lgb': [], 'xgb': [], 'ensemble': []}
    
    # Calculate scale_pos_weight
    pos_count = y_target.sum()
    neg_count = len(y_target) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    # Per-target hyperparameters
    if target == 'Other_Faults':
        # Tuned LGBM for Other_Faults
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'learning_rate': 0.01, 'n_estimators': 1000,
            'num_leaves': 15, 'max_depth': 5,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 5,
            'random_state': 42
        }
        # Tuned XGB for Other_Faults
        xgb_params = {
            'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 4,
            'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.7,
            'gamma': 0.05, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': 1.0, 'random_state': 42,
            'tree_method': 'hist', 'eval_metric': 'auc'
        }
    elif target == 'Bumps':
        # Standard params for Bumps
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'learning_rate': 0.05, 'n_estimators': 500,
            'num_leaves': 31, 'max_depth': -1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_samples': 20, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': scale_pos_weight, 'random_state': 42
        }
        xgb_params = {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.05, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': scale_pos_weight, 'random_state': 42,
            'tree_method': 'hist', 'eval_metric': 'auc'
        }
    elif target == 'K_Scratch':
        # XGB preferred for K_Scratch
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'learning_rate': 0.05, 'n_estimators': 500,
            'num_leaves': 31, 'max_depth': -1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_samples': 20, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': scale_pos_weight, 'random_state': 42
        }
        xgb_params = {
            'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 6,
            'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.05, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': scale_pos_weight, 'random_state': 42,
            'tree_method': 'hist', 'eval_metric': 'auc'
        }
    else:
        # LGBM preferred for Pastry, Z_Scratch, Stains, Dirtiness
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'learning_rate': 0.05, 'n_estimators': 500,
            'num_leaves': 31, 'max_depth': -1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_samples': 20, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': scale_pos_weight, 'random_state': 42
        }
        xgb_params = {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.05, 'reg_alpha': 0.05, 'reg_lambda': 0.8,
            'scale_pos_weight': scale_pos_weight, 'random_state': 42,
            'tree_method': 'hist', 'eval_metric': 'auc'
        }
    
    # Train 5-fold models
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_target)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_target[train_idx], y_target[val_idx]
        
        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        target_models['lgb'].append(lgb_model)
        fold_aucs['lgb'].append(roc_auc_score(y_val, lgb_pred))
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50, verbose=False)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        target_models['xgb'].append(xgb_model)
        fold_aucs['xgb'].append(roc_auc_score(y_val, xgb_pred))
        
        # Ensemble prediction
        ensemble_pred = (lgb_pred + xgb_pred) / 2.0
        fold_aucs['ensemble'].append(roc_auc_score(y_val, ensemble_pred))
    
    # Determine best model strategy based on CV performance
    avg_lgb = np.mean(fold_aucs['lgb'])
    avg_xgb = np.mean(fold_aucs['xgb'])
    avg_ensemble = np.mean(fold_aucs['ensemble'])
    
    if target == 'K_Scratch':
        best_auc = avg_xgb
        model_strategy = 'xgb'
    elif target == 'Bumps':
        best_auc = avg_ensemble
        model_strategy = 'ensemble'
    else:
        # Pastry, Z_Scratch, Stains, Dirtiness, Other_Faults use LGBM
        best_auc = avg_lgb
        model_strategy = 'lgb'
    
    final_models[target] = {
        'models': target_models,
        'strategy': model_strategy,
        'features': features
    }
    all_aucs[target] = best_auc
    print(f"  {target}: {best_auc:.4f} ({model_strategy})")

print("\n" + "="*30)
print("FINAL VALIDATION RESULTS")
print("="*30)
for t in targets:
    print(f"{t:15s}: {all_aucs[t]:.4f}")
print("-"*30)
mean_auc = np.mean(list(all_aucs.values()))
variance = np.var(list(all_aucs.values()))
min_auc = min(all_aucs.values())
print(f"Mean AUC       : {mean_auc:.4f}")
print(f"Variance       : {variance:.4f}")
print(f"Min AUC        : {min_auc:.4f}")

print("\nSaving models...")
with open('/workdir/model.pkl', 'wb') as f:
    pickle.dump({
        'models': final_models,
        'features': features,
        'targets': targets
    }, f)
print("Done.")
TRAINING_SCRIPT

cat > /workdir/predict.py << 'PREDICT_SCRIPT'
#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Feature engineering - MUST be identical to training"""
    df = df.copy()
    # Basic geometric features
    df['X_Range'] = df['X_Maximum'] - df['X_Minimum']
    df['Y_Range'] = df['Y_Maximum'] - df['Y_Minimum']
    df['Area_Perimeter_Ratio'] = df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'] + 1e-5)
    df['Luminosity_Range'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']
    df['Density'] = df['Pixels_Areas'] / (df['X_Range'] * df['Y_Range'] + 1e-5)
    df['Avg_Luminosity'] = df['Sum_of_Luminosity'] / (df['Pixels_Areas'] + 1e-5)
    df['Orientation_Luminosity'] = df['Orientation_Index'] * df['Luminosity_Index']
    df['Log_Area_Perimeter'] = np.log1p(df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'] + 1e-5))
    df['X_Center'] = (df['X_Maximum'] + df['X_Minimum']) / 2.0
    df['Y_Center'] = (df['Y_Maximum'] + df['Y_Minimum']) / 2.0
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Polynomial features for geometric columns (degree 2, interaction only)
    poly_cols = ['X_Range', 'Y_Range', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[poly_cols])
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    
    # Keep only interaction terms (those with ' ')
    new_cols = [c for c in poly_feature_names if ' ' in c]
    for i, col in enumerate(new_cols):
        df[col] = poly_features[:, poly_feature_names == col][:, 0]
    
    # Replace any new infinite values
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

# Load models
with open('/workdir/model.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    final_models = artifacts['models']
    features = artifacts['features']
    targets = artifacts['targets']

# Load test data
test_df = pd.read_csv(sys.argv[1])
known_targets = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
test_df = create_features(test_df)
X_test = test_df[features].values

predictions = {}

# Predict each target with appropriate model strategy
for target in targets:
    model_info = final_models[target]
    target_models = model_info['models']
    strategy = model_info['strategy']
    
    pred_sum = np.zeros(len(X_test))
    
    if strategy == 'ensemble':
        # Ensemble: average LGBM and XGBoost
        for fold in range(5):
            lgb_pred = target_models['lgb'][fold].predict_proba(X_test)[:, 1]
            xgb_pred = target_models['xgb'][fold].predict_proba(X_test)[:, 1]
            pred_sum += (lgb_pred + xgb_pred) / 2.0
        predictions[target] = pred_sum / 5.0
    
    elif strategy == 'xgb':
        # XGBoost only
        for fold in range(5):
            xgb_pred = target_models['xgb'][fold].predict_proba(X_test)[:, 1]
            pred_sum += xgb_pred
        predictions[target] = pred_sum / 5.0
    
    else:
        # LightGBM only (default)
        for fold in range(5):
            lgb_pred = target_models['lgb'][fold].predict_proba(X_test)[:, 1]
            pred_sum += lgb_pred
        predictions[target] = pred_sum / 5.0

# Create output
output_df = pd.DataFrame({'id': test_df['id']})
for target in targets:
    output_df[target] = predictions[target]

output_df.to_csv('/workdir/predictions.csv', index=False)
print("Predictions saved to /workdir/predictions.csv")
PREDICT_SCRIPT

chmod +x /workdir/predict.py
echo "✓ Solution ready!"
