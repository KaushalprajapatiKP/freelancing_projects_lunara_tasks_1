#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYTHON_EOF'
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

print("="*60, file=sys.stderr)
print("NIFTY Stock Movement Prediction - Multi-class Classification", file=sys.stderr)
print("="*60, file=sys.stderr)

# Custom transformer for date features
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            dates = pd.to_datetime(X['date'], errors='coerce')
        else:
            dates = pd.to_datetime(X, errors='coerce')
        
        df_dates = pd.DataFrame({
            'year': dates.dt.year,
            'month': dates.dt.month,
            'day': dates.dt.day,
            'dayofweek': dates.dt.dayofweek
        })
        return df_dates

# Load Data
print("\nStep 1: Loading data...", file=sys.stderr)
df = pd.read_csv('/app/data/train.csv')
print(f"  Loaded {len(df)} training samples", file=sys.stderr)
print(f"  Label distribution:\n{df['label'].value_counts()}", file=sys.stderr)

# Define Preprocessing
print("\nStep 2: Defining pipeline...", file=sys.stderr)

# Text pipeline - unigrams only for speed
text_features = 'news'
text_transformer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 1))

# Date pipeline
date_features = ['date']
date_transformer = DateFeatureExtractor()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features),
        ('date', date_transformer, date_features)
    ]
)

# Full Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# Training
print("\nStep 3: Training full model...", file=sys.stderr)
X = df[['news', 'date']]
y = df['label']
model.fit(X, y)

# Saving
print("\nStep 4: Saving model...", file=sys.stderr)
joblib.dump(model, '/app/model.pkl')
print(f"  ✓ Model saved to /app/model.pkl", file=sys.stderr)

print("\nStep 5: Creating predict.py...", file=sys.stderr)
predict_py_content = '''#!/usr/bin/env python3
import sys
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for date features (Must be defined for joblib load)
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            dates = pd.to_datetime(X['date'], errors='coerce')
        else:
            dates = pd.to_datetime(X, errors='coerce')
        
        df_dates = pd.DataFrame({
            'year': dates.dt.year,
            'month': dates.dt.month,
            'day': dates.dt.day,
            'dayofweek': dates.dt.dayofweek
        })
        return df_dates

if __name__ == '__main__':
    test_csv_path = sys.argv[1]

    model = joblib.load('/app/model.pkl')
    test_df = pd.read_csv(test_csv_path)

    predictions = model.predict(test_df)

    pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    }).to_csv('/app/predictions.csv', index=False)
    print("Predictions saved.", file=sys.stderr)
'''

with open('/app/predict.py', 'w') as f:
    f.write(predict_py_content)

import os
os.chmod('/app/predict.py', 0o755)
print(f"  ✓ Created /app/predict.py", file=sys.stderr)

print("\n" + "="*60, file=sys.stderr)
print("Solution completed successfully!", file=sys.stderr)
print("="*60, file=sys.stderr)

PYTHON_EOF
