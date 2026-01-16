#!/usr/bin/env bash
set -euo pipefail

# Copy this script to /workdir/solution.sh for grader (using $0 to avoid permission issues)
if [ ! -f /workdir/solution.sh ]; then
    cp "$0" /workdir/solution.sh 2>/dev/null || true
    chmod +x /workdir/solution.sh 2>/dev/null || true
fi

# Create model_utils.py with feature engineering and ensemble class
cat > /workdir/model_utils.py << 'MODEL_UTILS_EOF'
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X):
        for col in X.columns:
            le = LabelEncoder()
            unique_vals = X[col].fillna('missing').astype(str).unique().tolist()
            unique_vals.append('unknown')
            le.fit(unique_vals)
            self.encoders[col] = le
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encoders:
                X_encoded[col] = X_encoded[col].fillna('missing').astype(str)
                known_categories = set(self.encoders[col].classes_)
                X_encoded[col] = X_encoded[col].apply(
                    lambda x: x if x in known_categories else 'unknown'
                )
                X_encoded[col] = self.encoders[col].transform(X_encoded[col])
            else:
                X_encoded[col] = 0
        return X_encoded
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class LGBMWrapper:
    """Wrapper for LightGBM model to make it compatible with ensemble"""
    def __init__(self, model):
        self.model = model
    
    def predict_proba(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration if hasattr(self.model, 'best_iteration') else None)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict_proba(self, X):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X)
            predictions.append(pred * weight)
        return np.sum(predictions, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def create_features(df):
    """Create engineered features from the raw data"""
    df_feat = df.copy()
    
    # 1. Player career features
    player_stats = df_feat.groupby('player_name').agg({
        'pts': ['mean', 'std', 'max', 'min'],
        'reb': ['mean', 'std', 'max', 'min'],
        'ast': ['mean', 'std', 'max', 'min'],
        'gp': ['sum', 'mean'],
        'age': ['min', 'max'],
        'net_rating': ['mean', 'std'],
        'usg_pct': ['mean', 'std'],
        'ts_pct': ['mean', 'std'],
        'oreb_pct': ['mean'],
        'dreb_pct': ['mean'],
        'ast_pct': ['mean']
    })
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns.values]
    player_stats = player_stats.reset_index()
    df_feat = df_feat.merge(player_stats, on='player_name', how='left', suffixes=('', '_career'))
    
    # 2. Team features
    team_stats = df_feat.groupby(['team_abbreviation', 'season']).agg({
        'pts': ['mean', 'std'],
        'reb': ['mean', 'std'],
        'ast': ['mean', 'std'],
        'net_rating': ['mean', 'std'],
        'gp': 'mean'
    })
    team_stats.columns = ['_'.join(col).strip() + '_team' for col in team_stats.columns.values]
    team_stats = team_stats.reset_index()
    df_feat = df_feat.merge(team_stats, on=['team_abbreviation', 'season'], how='left')
    
    # 3. Draft features
    df_feat['is_undrafted'] = (df_feat['draft_year'] == 'Undrafted').astype(int)
    df_feat['draft_year_numeric'] = pd.to_numeric(df_feat['draft_year'], errors='coerce')
    df_feat['draft_round_numeric'] = pd.to_numeric(df_feat['draft_round'], errors='coerce')
    df_feat['draft_number_numeric'] = pd.to_numeric(df_feat['draft_number'], errors='coerce')
    df_feat['season_year'] = df_feat['season'].str[:4].astype(int)
    df_feat['years_since_draft'] = df_feat['season_year'] - df_feat['draft_year_numeric']
    
    # 4. Physical features
    df_feat['bmi'] = df_feat['player_weight'] / ((df_feat['player_height'] / 100) ** 2)
    df_feat['height_weight_ratio'] = df_feat['player_height'] / df_feat['player_weight']
    
    # 5. Performance ratios
    df_feat['pts_per_gp'] = df_feat['pts'] / (df_feat['gp'] + 1)
    df_feat['reb_per_gp'] = df_feat['reb'] / (df_feat['gp'] + 1)
    df_feat['ast_per_gp'] = df_feat['ast'] / (df_feat['gp'] + 1)
    df_feat['pts_reb_ratio'] = df_feat['pts'] / (df_feat['reb'] + 1)
    df_feat['pts_ast_ratio'] = df_feat['pts'] / (df_feat['ast'] + 1)
    df_feat['reb_ast_ratio'] = df_feat['reb'] / (df_feat['ast'] + 1)
    df_feat['total_production'] = df_feat['pts'] + df_feat['reb'] + df_feat['ast']
    
    # 6. Efficiency features
    df_feat['offensive_efficiency'] = df_feat['pts'] * df_feat['ts_pct']
    df_feat['usage_efficiency'] = df_feat['usg_pct'] * df_feat['ts_pct']
    
    # 7. Age-based features
    df_feat['age_squared'] = df_feat['age'] ** 2
    df_feat['is_rookie'] = (df_feat['years_since_draft'] <= 1).astype(int)
    df_feat['is_veteran'] = (df_feat['years_since_draft'] >= 10).astype(int)
    df_feat['is_prime'] = ((df_feat['age'] >= 25) & (df_feat['age'] <= 30)).astype(int)
    
    # 8. Position proxy features
    df_feat['guard_proxy'] = df_feat['ast'] / (df_feat['reb'] + 1)
    df_feat['center_proxy'] = df_feat['reb'] / (df_feat['ast'] + 1)
    
    # 9. College features
    df_feat['has_college'] = (~df_feat['college'].isna()).astype(int)
    college_counts = df_feat['college'].value_counts().to_dict()
    df_feat['college_player_count'] = df_feat['college'].map(college_counts).fillna(0)
    
    # 10. Country features
    df_feat['is_international'] = (df_feat['country'] != 'USA').astype(int)
    
    # 11. Advanced interaction features
    df_feat['performance_consistency'] = df_feat['pts_std'] / (df_feat['pts_mean'] + 1)
    df_feat['reb_consistency'] = df_feat['reb_std'] / (df_feat['reb_mean'] + 1)
    df_feat['ast_consistency'] = df_feat['ast_std'] / (df_feat['ast_mean'] + 1)
    df_feat['pts_vs_team_avg'] = df_feat['pts'] - df_feat['pts_mean_team']
    df_feat['reb_vs_team_avg'] = df_feat['reb'] - df_feat['reb_mean_team']
    df_feat['ast_vs_team_avg'] = df_feat['ast'] - df_feat['ast_mean_team']
    df_feat['high_usage_efficiency'] = (df_feat['usg_pct'] > 0.25) * df_feat['ts_pct']
    df_feat['low_usage_efficiency'] = (df_feat['usg_pct'] <= 0.15) * df_feat['ts_pct']
    df_feat['gp_ratio'] = df_feat['gp'] / 82
    df_feat['is_regular'] = (df_feat['gp'] >= 50).astype(int)
    df_feat['is_bench'] = (df_feat['gp'] < 30).astype(int)
    df_feat['scoring_diversity'] = df_feat['ast'] / (df_feat['pts'] + 1)
    df_feat['rebounding_rate'] = (df_feat['oreb_pct'] + df_feat['dreb_pct']) / 2
    df_feat['career_improvement'] = df_feat['pts'] - df_feat['pts_mean']
    df_feat['career_peak'] = (df_feat['pts'] == df_feat['pts_max']).astype(int)
    
    df_feat = df_feat.drop(columns=['season_year'], errors='ignore')
    return df_feat
MODEL_UTILS_EOF

# Main training script
python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add /workdir to path to find model_utils
import sys
sys.path.insert(0, '/workdir')

from model_utils import MultiColumnLabelEncoder, EnsembleModel, LGBMWrapper, create_features

print("\n" + "="*80, file=sys.stderr)
print("NBA Player Performance Bucket Prediction - Ensemble Solution", file=sys.stderr)
print("="*80, file=sys.stderr)

TRAIN_PATH = Path("/workdir/data/train.csv")
MODEL_PATH = Path("/workdir/model.pkl")
OUTDIR = Path("/workdir/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("\n[1/5] Loading data...", file=sys.stderr)
train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
print(f"  Train: {len(train_df)} samples", file=sys.stderr)

if 'performance_bucket' not in train_df.columns:
    print("ERROR: 'performance_bucket' column not found", file=sys.stderr)
    sys.exit(1)

y_train = train_df['performance_bucket'].values
n_classes = len(np.unique(y_train))
print(f"  Classes: {n_classes} (0-{n_classes-1})", file=sys.stderr)

print("\n[2/5] Feature engineering...", file=sys.stderr)
train_df_eng = create_features(train_df)

drop_cols = ['id', 'player_name', 'team_abbreviation', 'college', 'country', 'season',
             'draft_year', 'draft_round', 'draft_number', 'performance_metric', 'performance_bucket']
feature_cols_eng = [col for col in train_df_eng.columns if col not in drop_cols]

numeric_features = [col for col in feature_cols_eng if train_df_eng[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in feature_cols_eng if col not in numeric_features]

print(f"  Total features: {len(feature_cols_eng)}")
print(f"  Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}", file=sys.stderr)

print("\n[3/5] Preprocessing...", file=sys.stderr)
# Create numeric transformer pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numeric_transformer = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features)
], remainder='drop')

cat_encoder = MultiColumnLabelEncoder()
X_train_cat = train_df_eng[categorical_features] if categorical_features else pd.DataFrame()
X_train_num = train_df_eng[numeric_features]

if len(categorical_features) > 0:
    X_train_cat_encoded = cat_encoder.fit_transform(X_train_cat)
else:
    X_train_cat_encoded = pd.DataFrame()

X_train_num_processed = numeric_transformer.fit_transform(X_train_num)
X_train_processed = np.hstack([X_train_num_processed, X_train_cat_encoded.values]) if len(categorical_features) > 0 else X_train_num_processed

print(f"  Final feature matrix: {X_train_processed.shape}", file=sys.stderr)

print("\n[4/5] Training models...", file=sys.stderr)

# LightGBM
lgb_params = {
    'objective': 'multiclass',
    'num_class': 50,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'max_depth': 12,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'lambda_l1': 0.05,
    'lambda_l2': 0.1,
    'n_estimators': 1500,
    'random_state': SEED,
    'verbose': -1,
    'min_gain_to_split': 0.01,
    'max_bin': 255
}

print("  Training LightGBM...", file=sys.stderr)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
lgb_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed, y_train)):
    X_tr, X_val = X_train_processed[train_idx], X_train_processed[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_val, y_pred_class, average='macro')
    lgb_f1_scores.append(f1)

print(f"  LightGBM CV Macro F1: {np.mean(lgb_f1_scores):.4f}", file=sys.stderr)

lgb_train_full = lgb.Dataset(X_train_processed, label=y_train)
lgb_model_raw = lgb.train(lgb_params, lgb_train_full, num_boost_round=1500)
lgb_model_final = LGBMWrapper(lgb_model_raw)

# XGBoost
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 50,
    'max_depth': 10,
    'learning_rate': 0.03,
    'n_estimators': 1200,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'min_child_weight': 2,
    'gamma': 0.1
}

print("  Training XGBoost...", file=sys.stderr)
xgb_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed, y_train)):
    X_tr, X_val = X_train_processed[train_idx], X_train_processed[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    
    y_pred = xgb_model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    xgb_f1_scores.append(f1)

print(f"  XGBoost CV Macro F1: {np.mean(xgb_f1_scores):.4f}", file=sys.stderr)

xgb_model_final = xgb.XGBClassifier(**xgb_params)
xgb_model_final.fit(X_train_processed, y_train)

# Random Forest
print("  Training Random Forest...", file=sys.stderr)
rf_params = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': SEED,
    'n_jobs': -1,
    'class_weight': 'balanced_subsample'
}
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train_processed, y_train)

# Create ensemble
print("\n[5/5] Creating ensemble and saving...", file=sys.stderr)
ensemble_model = EnsembleModel([lgb_model_final, xgb_model_final, rf_model], weights=[0.5, 0.35, 0.15])

model_artifacts = {
    'ensemble_model': ensemble_model,
    'lgb_model': lgb_model_final,
    'xgb_model': xgb_model_final,
    'rf_model': rf_model,
    'numeric_transformer': numeric_transformer,
    'cat_encoder': cat_encoder,
    'feature_cols_eng': feature_cols_eng,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}

joblib.dump(model_artifacts, MODEL_PATH)
joblib.dump(model_artifacts, OUTDIR / 'model.pkl')

print(f"  Model saved to {MODEL_PATH}", file=sys.stderr)
print("\n" + "="*80, file=sys.stderr)
print("Training Complete!", file=sys.stderr)
print("="*80, file=sys.stderr)

PYTHON_EOF

# Generate predict.py
cat > /workdir/predict.py << 'PREDICT_SCRIPT'
#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add /workdir to path to find model_utils
sys.path.insert(0, '/workdir')

from model_utils import create_features, LGBMWrapper, EnsembleModel, MultiColumnLabelEncoder

# Make custom classes available in __main__ namespace for unpickling
import __main__
__main__.LGBMWrapper = LGBMWrapper
__main__.EnsembleModel = EnsembleModel
__main__.MultiColumnLabelEncoder = MultiColumnLabelEncoder

def main():
    if len(sys.argv) < 3:
        print("Usage: predict.py <model_path> <test_csv_path>", file=sys.stderr)
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_csv_path = sys.argv[2]
    output_path = '/workdir/outputs/predictions.csv'
    
    print(f"Loading model from {model_path}...", file=sys.stderr)
    artifacts = joblib.load(model_path)
    
    ensemble_model = artifacts['ensemble_model']
    numeric_transformer = artifacts['numeric_transformer']
    cat_encoder = artifacts['cat_encoder']
    feature_cols_eng = artifacts['feature_cols_eng']
    numeric_features = artifacts['numeric_features']
    categorical_features = artifacts['categorical_features']
    
    print(f"Loading test data from {test_csv_path}...", file=sys.stderr)
    test_df = pd.read_csv(test_csv_path, low_memory=False)
    print(f"  Test samples: {len(test_df)}", file=sys.stderr)
    
    print("Engineering features...", file=sys.stderr)
    test_df_eng = create_features(test_df)
    
    X_test_num = test_df_eng[numeric_features]
    X_test_num_processed = numeric_transformer.transform(X_test_num)
    
    if len(categorical_features) > 0:
        X_test_cat = test_df_eng[categorical_features]
        X_test_cat_encoded = cat_encoder.transform(X_test_cat)
        X_test_processed = np.hstack([X_test_num_processed, X_test_cat_encoded.values])
    else:
        X_test_processed = X_test_num_processed
    
    print("Generating predictions...", file=sys.stderr)
    predictions = ensemble_model.predict(X_test_processed)
    predictions = np.clip(predictions, 0, 49)
    
    print(f"Saving predictions to {output_path}...", file=sys.stderr)
    pd.DataFrame({
        'id': test_df['id'].astype(str),
        'prediction': predictions
    }).to_csv(output_path, index=False)
    
    print("Done.", file=sys.stderr)

if __name__ == '__main__':
    main()
PREDICT_SCRIPT

chmod +x /workdir/predict.py
echo "✓ Solution complete" >&2
