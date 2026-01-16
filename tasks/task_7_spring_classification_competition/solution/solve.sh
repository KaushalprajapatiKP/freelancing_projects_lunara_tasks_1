#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60, file=sys.stderr)
print("Stroke Classification - XGBoost Solution", file=sys.stderr)
print("="*60, file=sys.stderr)

print("\nStep 1: Loading training data...", file=sys.stderr)
df = pd.read_csv('/app/data/train.csv')
print(f"  Loaded {len(df)} training samples", file=sys.stderr)

print("\nStep 2: Preparing features and target...", file=sys.stderr)
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

print(f"  Features: {len(X.columns)}", file=sys.stderr)
print(f"  Target distribution: {y.value_counts().to_dict()}", file=sys.stderr)

print("\nStep 3: Defining feature types...", file=sys.stderr)
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
print(f"  Numeric features: {numeric_features}", file=sys.stderr)
print(f"  Categorical features: {categorical_features}", file=sys.stderr)

print("\nStep 4: Creating preprocessing pipeline...", file=sys.stderr)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep hypertension and heart_disease
)

print("\nStep 5: Creating model with tuned hyperparameters...", file=sys.stderr)
# Calculate scale_pos_weight
neg, pos = np.bincount(y)
scale_weight = neg / pos
print(f"  Class imbalance ratio: {scale_weight:.2f}", file=sys.stderr)
print(f"  Using scale_pos_weight: 5 (tuned)", file=sys.stderr)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    scale_pos_weight=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("\nStep 6: Cross-validation...", file=sys.stderr)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
print(f"  Cross-validation F1 scores: {scores}", file=sys.stderr)
print(f"  Mean F1: {scores.mean():.4f}", file=sys.stderr)

print("\nStep 7: Training final model on full data...", file=sys.stderr)
pipeline.fit(X, y)

print("\nStep 8: Saving model...", file=sys.stderr)
joblib.dump(pipeline, '/app/model_pipeline.joblib')
print(f"  ✓ Model saved to /app/model_pipeline.joblib", file=sys.stderr)

print("\nStep 9: Creating predict.py...", file=sys.stderr)
predict_py_content = '''#!/usr/bin/env python3
import sys
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print('Usage: python3 predict.py <test_csv_path>', file=sys.stderr)
        sys.exit(1)

    test_path = sys.argv[1]

    # Load data
    try:
        df = pd.read_csv(test_path)
    except Exception as e:
        print(f'Error reading file: {e}', file=sys.stderr)
        sys.exit(1)

    # Save IDs for output
    if 'id' in df.columns:
        ids = df['id']
        X_test = df.drop(columns=['id'], errors='ignore')
    elif 'ID' in df.columns:
        ids = df['ID']
        X_test = df.drop(columns=['ID'], errors='ignore')
    else:
        print("Error: 'id' or 'ID' column not found in test data", file=sys.stderr)
        sys.exit(1)

    # Load model
    try:
        pipeline = joblib.load('/app/model_pipeline.joblib')
    except Exception as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        sys.exit(1)

    # Predict probabilities (class 1 probability is at index 1)
    try:
        probs = pipeline.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f'Error during prediction: {e}', file=sys.stderr)
        sys.exit(1)

    # Create output dataframe
    output = pd.DataFrame({
        'ID': ids,
        'TARGET': probs
    })

    # Save to csv
    output.to_csv('/app/predictions.csv', index=False)
    print('Predictions saved to /app/predictions.csv', file=sys.stderr)

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
print(f"Mean CV F1-Score: {scores.mean():.4f}", file=sys.stderr)
print("="*60, file=sys.stderr)

PYTHON_EOF

echo "Solution complete! Created:"
echo "  - /app/model_pipeline.joblib (trained model)"
echo "  - /app/predict.py (prediction script)"
