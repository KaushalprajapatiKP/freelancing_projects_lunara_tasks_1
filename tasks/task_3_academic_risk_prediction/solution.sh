#!/usr/bin/env bash
set -euo pipefail

# Create utils.py with custom classes
cat > /workdir/utils.py << 'UTILS_SCRIPT'
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, LabelBinarizer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # === BASE RATIO FEATURES ===
        df["approval_rate_1st"] = df["Curricular units 1st sem (approved)"] / (df["Curricular units 1st sem (enrolled)"] + 1)
        df["approval_rate_2nd"] = df["Curricular units 2nd sem (approved)"] / (df["Curricular units 2nd sem (enrolled)"] + 1)
        df["eval_rate_1st"] = df["Curricular units 1st sem (evaluations)"] / (df["Curricular units 1st sem (enrolled)"] + 1)
        df["eval_rate_2nd"] = df["Curricular units 2nd sem (evaluations)"] / (df["Curricular units 2nd sem (enrolled)"] + 1)
        
        # === SEMESTER DELTA FEATURES ===
        df["grade_diff"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
        df["enrolled_diff"] = df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 1st sem (enrolled)"]
        df["approved_diff"] = df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]
        
        # === AGGREGATE FEATURES ===
        df["total_approved"] = df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
        df["total_enrolled"] = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
        df["overall_approval_rate"] = df["total_approved"] / (df["total_enrolled"] + 1)
        df["grade_avg"] = (df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]) / 2

        # === KEY FEATURES ===
        df["grade_ratio"] = df["Curricular units 2nd sem (grade)"] / (df["Curricular units 1st sem (grade)"] + 1)
        df["approval_ratio_change"] = df["approval_rate_2nd"] - df["approval_rate_1st"]
        
        if "Admission grade" in df.columns and "Previous qualification (grade)" in df.columns:
            df["grade_diff_admission"] = df["Admission grade"] - df["Previous qualification (grade)"]
        
        # === INTERACTION FEATURES ===
        if "Scholarship holder" in df.columns and "Debtor" in df.columns:
            df["scholarship_debtor"] = df["Scholarship holder"] * df["Debtor"]
        
        if "Tuition fees up to date" in df.columns and "Scholarship holder" in df.columns:
            df["tuition_scholarship"] = df["Tuition fees up to date"] * df["Scholarship holder"]
        
        # Replace infinity and NaN
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Clip extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'id' and len(df) > 0:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q99)
        
        return df

class MulticlassTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smooth='auto'):
        self.columns = columns
        self.smooth = smooth
        self.encoders = []
        self.lb = LabelBinarizer()
        self.classes_ = None
    
    def fit(self, X, y):
        if self.columns is None:
            self.columns = X.columns.tolist()
        
        y_onehot = self.lb.fit_transform(y)
        self.classes_ = self.lb.classes_
        
        self.encoders = []
        for i in range(len(self.classes_)):
            te = TargetEncoder(smooth=self.smooth, target_type='binary')
            te.fit(X[self.columns], y_onehot[:, i])
            self.encoders.append(te)
        
        return self
    
    def transform(self, X):
        output_dfs = []
        for i, te in enumerate(self.encoders):
            out = te.transform(X[self.columns])
            cols = [f"{col}_target_{self.classes_[i]}" for col in self.columns]
            output_dfs.append(pd.DataFrame(out, columns=cols, index=X.index))
        
        return pd.concat(output_dfs, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.columns
        
        feature_names = []
        for i, class_label in enumerate(self.classes_):
            feature_names.extend([f"{col}_target_{class_label}" for col in self.columns])
        return np.array(feature_names)
    
    def set_output(self, *, transform=None):
        return super().set_output(transform=transform) if hasattr(super(), 'set_output') else self
UTILS_SCRIPT

# Training script
python3 << 'TRAINING_SCRIPT'
import sys
sys.path.insert(0, '/workdir')
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from utils import FeatureEngineer, MulticlassTargetEncoder
import warnings
warnings.filterwarnings('ignore')

# Set seeds for determinism
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)

print("Loading data...")
train_df = pd.read_csv("/workdir/data/train.csv")

# Check for duplicate IDs
if train_df['id'].duplicated().any():
    print(f"Warning: {train_df['id'].duplicated().sum()} duplicate IDs found, keeping first")
    train_df = train_df.drop_duplicates(subset=['id'], keep='first')

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(train_df["Target"])
class_names = le.classes_

print(f"Target mapping: {dict(zip(class_names, range(len(class_names))))}")

# Prepare features (drop id and Target)
X = train_df.drop(columns=["id", "Target"])

# Define categorical columns for target encoding
cat_cols = ['Course', 'Nacionality', "Mother's qualification", "Father's qualification", 
            "Mother's occupation", "Father's occupation", 'Previous qualification']

# Build pipeline
feature_eng = FeatureEngineer()
target_enc = MulticlassTargetEncoder(columns=cat_cols, smooth='auto')

# ColumnTransformer: apply target encoding to categoricals, passthrough for others
preprocessor = ColumnTransformer(
    transformers=[
        ('target_enc', target_enc, cat_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Create full pipeline
full_pipeline = Pipeline([
    ('feature_eng', feature_eng),
    ('preprocessor', preprocessor)
])

# Transform data
print("Transforming features...")
X_transformed = full_pipeline.fit_transform(X, y_encoded)

# Convert to DataFrame if needed
if isinstance(X_transformed, np.ndarray):
    # Get feature names
    feature_names = []
    # Target encoded features
    for col in cat_cols:
        for cls in class_names:
            feature_names.append(f"{col}_target_{cls}")
    # Passthrough features (all original + engineered)
    all_cols = list(X.columns) + [col for col in feature_eng.transform(X).columns if col not in X.columns]
    passthrough_cols = [col for col in all_cols if col not in cat_cols]
    feature_names.extend(passthrough_cols)
    
    # Create DataFrame
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names[:X_transformed.shape[1]])

print(f"Transformed features shape: {X_transformed.shape}")

# Cross-validation
print("\nRunning Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = {'acc': [], 'f1_macro': [], 'f1_enrolled': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y_encoded), 1):
    X_train_fold, X_val_fold = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
    
    # Base models
    xgb = XGBClassifier(
        objective="multi:softprob", n_estimators=1200, learning_rate=0.04,
        max_depth=8, min_child_weight=4, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.4, reg_lambda=1.5, gamma=0.15, random_state=42, n_jobs=-1,
        eval_metric="mlogloss", verbosity=0
    )
    
    lgb = LGBMClassifier(
        objective="multiclass", n_estimators=1200, learning_rate=0.04,
        max_depth=8, num_leaves=60, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.3, reg_lambda=1.0, min_child_samples=20, random_state=42,
        n_jobs=-1, verbose=-1
    )
    
    hgb = HistGradientBoostingClassifier(
        max_iter=1200, learning_rate=0.04, max_depth=8,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42, verbose=0
    )
    
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    
    # Voting classifier
    voting = VotingClassifier(
        estimators=[('xgb', xgb), ('lgb', lgb), ('hgb', hgb), ('rf', rf)],
        voting='soft'
    )
    
    voting.fit(X_train_fold, y_train_fold)
    
    # Evaluate
    y_pred = voting.predict(X_val_fold)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_val_fold, y_pred)
    f1_macro = f1_score(y_val_fold, y_pred, average='macro')
    f1_per_class = f1_score(y_val_fold, y_pred, average=None)
    enrolled_idx = np.where(class_names == 'Enrolled')[0][0] if 'Enrolled' in class_names else 1
    f1_enrolled = f1_per_class[enrolled_idx]
    
    fold_scores['acc'].append(acc)
    fold_scores['f1_macro'].append(f1_macro)
    fold_scores['f1_enrolled'].append(f1_enrolled)
    
    print(f"Fold {fold}: Acc={acc:.4f}, Macro F1={f1_macro:.4f}, Enrolled F1={f1_enrolled:.4f}")

print(f"\nAverage Scores:")
print(f"Accuracy: {np.mean(fold_scores['acc']):.4f}")
print(f"Macro F1: {np.mean(fold_scores['f1_macro']):.4f}")
print(f"Enrolled F1: {np.mean(fold_scores['f1_enrolled']):.4f}")

# Train on full data
print("\nTraining on FULL data...")
xgb_final = XGBClassifier(
    objective="multi:softprob", n_estimators=1200, learning_rate=0.04,
    max_depth=8, min_child_weight=4, subsample=0.8, colsample_bytree=0.75,
    reg_alpha=0.4, reg_lambda=1.5, gamma=0.15, random_state=42, n_jobs=-1,
    eval_metric="mlogloss", verbosity=0
)

lgb_final = LGBMClassifier(
    objective="multiclass", n_estimators=1200, learning_rate=0.04,
    max_depth=8, num_leaves=60, subsample=0.8, colsample_bytree=0.75,
    reg_alpha=0.3, reg_lambda=1.0, min_child_samples=20, random_state=42,
    n_jobs=-1, verbose=-1
)

hgb_final = HistGradientBoostingClassifier(
    max_iter=1200, learning_rate=0.04, max_depth=8,
    min_samples_leaf=20, l2_regularization=0.1, random_state=42, verbose=0
)

rf_final = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)

voting_final = VotingClassifier(
    estimators=[('xgb', xgb_final), ('lgb', lgb_final), ('hgb', hgb_final), ('rf', rf_final)],
    voting='soft'
)

voting_final.fit(X_transformed, y_encoded)

# Save model
print("\nSaving model...")
with open('/workdir/model.pkl', 'wb') as f:
    pickle.dump({
        'model': voting_final,
        'pipeline': full_pipeline,
        'le': le,
        'class_names': class_names,
        'feature_cols': list(X_transformed.columns)
    }, f)

print("Done.")
TRAINING_SCRIPT

# Create predict.py script
cat > /workdir/predict.py << 'PREDICT_SCRIPT'
import sys
sys.path.insert(0, '/workdir')
import pandas as pd
import pickle
import numpy as np
from utils import FeatureEngineer, MulticlassTargetEncoder

if len(sys.argv) < 3:
    raise ValueError("Usage: predict.py <model_path> <test_csv_path>")

model_path = sys.argv[1]
test_csv_path = sys.argv[2]

# Load model
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    pipeline = model_data['pipeline']
    le = model_data['le']
    class_names = model_data['class_names']
    feature_cols = model_data['feature_cols']

# Load test data
test_df = pd.read_csv(test_csv_path)

# Store IDs
test_ids = test_df['id'].copy()

# Drop id and Target (if present) before processing
X_test = test_df.drop(columns=['id'], errors='ignore')
if 'Target' in X_test.columns:
    X_test = X_test.drop(columns=['Target'])

# Transform using pipeline
X_test_transformed = pipeline.transform(X_test)

# Convert to DataFrame if needed
if isinstance(X_test_transformed, np.ndarray):
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_cols[:X_test_transformed.shape[1]])

# Ensure all feature columns exist
for col in feature_cols:
    if col not in X_test_transformed.columns:
        X_test_transformed[col] = 0

# Select only the features used in training
X_test_transformed = X_test_transformed[feature_cols]

# Get probabilities
probs = model.predict_proba(X_test_transformed)

# Boost Enrolled class probability (index 1) by 1.1x
enrolled_idx = np.where(class_names == 'Enrolled')[0][0] if 'Enrolled' in class_names else 1
probs[:, enrolled_idx] *= 1.1

# Normalize probabilities
probs = probs / probs.sum(axis=1, keepdims=True)

# Predict
predictions_encoded = np.argmax(probs, axis=1)
predictions = le.inverse_transform(predictions_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': test_ids,
    'Target': predictions
})

output_df.to_csv('/workdir/predictions.csv', index=False)
print("Predictions saved to /workdir/predictions.csv")
PREDICT_SCRIPT

chmod +x /workdir/predict.py
echo "✓ Solution ready: solution.sh executed, model.pkl and predict.py created"
