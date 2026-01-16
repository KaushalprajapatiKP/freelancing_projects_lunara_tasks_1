#!/usr/bin/env bash
set -euo pipefail

# Copy this script to /workdir/solution.sh for grader (using $0 to avoid permission issues)
if [ ! -f /workdir/solution.sh ]; then
    cp "$0" /workdir/solution.sh 2>/dev/null || true
    chmod +x /workdir/solution.sh 2>/dev/null || true
fi

python3 << 'PYTHON_EOF'
import sys

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgbm
    HAS_LGBM = True
except:
    HAS_LGBM = False

print("="*80, file=sys.stderr)
print("Spotify Duration Prediction - Compliant Version", file=sys.stderr)
print("="*80, file=sys.stderr)

OUTDIR = Path("/workdir/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)

print("\n[1/3] Loading & analyzing training data...", file=sys.stderr)
train = pd.read_csv("/workdir/data/train.csv")
y = train['duration_bucket'].values

# Advanced target encoding with smoothing to prevent overfitting
print("  Computing smoothed target encoding patterns from training...", file=sys.stderr)

global_mean = train['duration_bucket'].mean()
smoothing_factor = 3  # Lower smoothing = more trust in data, better performance

# Track-level encoding with smoothing
track_counts = train.groupby('track_uri').size()
track_bucket_mean = train.groupby('track_uri')['duration_bucket'].mean()
track_bucket_std = train.groupby('track_uri')['duration_bucket'].std().fillna(0)
# Smoothed: (count * mean + smoothing * global_mean) / (count + smoothing)
track_bucket_smooth = (track_counts * track_bucket_mean + smoothing_factor * global_mean) / (track_counts + smoothing_factor)

# Artist-level encoding with smoothing
artist_counts = train.groupby('artist').size()
artist_bucket_mean = train.groupby('artist')['duration_bucket'].mean()
artist_bucket_std = train.groupby('artist')['duration_bucket'].std().fillna(0)
artist_bucket_smooth = (artist_counts * artist_bucket_mean + smoothing_factor * global_mean) / (artist_counts + smoothing_factor)

# Album-level encoding with smoothing
album_counts = train.groupby('album').size()
album_bucket_mean = train.groupby('album')['duration_bucket'].mean()
album_bucket_smooth = (album_counts * album_bucket_mean + smoothing_factor * global_mean) / (album_counts + smoothing_factor)

# Platform patterns
platform_bucket = train.groupby('platform')['duration_bucket'].mean()

# Reason patterns
reason_end_bucket = train.groupby('reason_end')['duration_bucket'].mean()
reason_start_bucket = train.groupby('reason_start')['duration_bucket'].mean()
reason_combo = train['reason_start'].fillna('u') + '_' + train['reason_end'].fillna('u')
train['reason_combo'] = reason_combo
reason_combo_bucket = train.groupby('reason_combo')['duration_bucket'].mean()

# Skip patterns
skip_bucket = train.groupby('skipped')['duration_bucket'].mean()

# Shuffle patterns
shuffle_bucket = train.groupby('shuffle')['duration_bucket'].mean()

# Frequency encodings (how common is this track/artist/album)
track_freq = train['track_uri'].value_counts().to_dict()
artist_freq = train['artist'].value_counts().to_dict()
album_freq = train['album'].value_counts().to_dict()

stats = {
    'track_bucket': track_bucket_smooth.to_dict(),
    'track_bucket_raw': track_bucket_mean.to_dict(),  # For high-count tracks
    'track_bucket_std': track_bucket_std.to_dict(),
    'track_bucket_count': track_counts.to_dict(),
    'track_freq': track_freq,
    'artist_bucket': artist_bucket_smooth.to_dict(),
    'artist_bucket_raw': artist_bucket_mean.to_dict(),
    'artist_bucket_std': artist_bucket_std.to_dict(),
    'artist_bucket_count': artist_counts.to_dict(),
    'artist_freq': artist_freq,
    'album_bucket': album_bucket_smooth.to_dict(),
    'album_freq': album_freq,
    'reason_end_bucket': reason_end_bucket.to_dict(),
    'reason_start_bucket': reason_start_bucket.to_dict(),
    'reason_combo_bucket': reason_combo_bucket.to_dict(),
    'skip_bucket': skip_bucket.to_dict(),
    'shuffle_bucket': shuffle_bucket.to_dict(),
    'platform_bucket': platform_bucket.to_dict(),
    'global_mean': global_mean,
    'smoothing_factor': smoothing_factor
}

print(f"  Learned patterns from {len(train)} samples", file=sys.stderr)
print(f"    Tracks: {len(track_bucket_mean)}", file=sys.stderr)
print(f"    Artists: {len(artist_bucket_mean)}", file=sys.stderr)

print("\n[2/3] Engineering features...", file=sys.stderr)

def make_features(df, stats):
    """Extract powerful features WITHOUT using duration_ms - Optimized for 0.55 Macro F1"""
    gm = stats['global_mean']
    
    # Smoothed target encodings (use raw for high-count, smoothed for low-count)
    track_count = df['track_uri'].map(stats['track_bucket_count']).fillna(0)
    track_tgt_smooth = df['track_uri'].map(stats['track_bucket']).fillna(gm)
    track_tgt_raw = df['track_uri'].map(stats['track_bucket_raw']).fillna(gm)
    # Use raw for tracks seen 3+ times, smoothed otherwise (more aggressive)
    track_tgt = np.where(track_count >= 3, track_tgt_raw, track_tgt_smooth)
    track_std = df['track_uri'].map(stats['track_bucket_std']).fillna(0)
    track_freq = df['track_uri'].map(stats['track_freq']).fillna(0)
    
    # Artist encoding
    artist_count = df['artist'].map(stats.get('artist_bucket_count', {})).fillna(0)
    artist_tgt_smooth = df['artist'].map(stats['artist_bucket']).fillna(gm)
    artist_tgt_raw = df['artist'].map(stats['artist_bucket_raw']).fillna(gm)
    artist_tgt = np.where(artist_count >= 3, artist_tgt_raw, artist_tgt_smooth)
    artist_std = df['artist'].map(stats['artist_bucket_std']).fillna(0)
    artist_freq = df['artist'].map(stats['artist_freq']).fillna(0)
    
    # Album encoding
    album_tgt = df['album'].map(stats['album_bucket']).fillna(gm)
    album_freq = df['album'].map(stats['album_freq']).fillna(0)
    
    # Platform patterns
    platform_tgt = df['platform'].map(stats['platform_bucket']).fillna(gm)
    
    # Reason patterns (strong behavioral signals)
    reason_end_tgt = df['reason_end'].map(stats['reason_end_bucket']).fillna(gm)
    reason_start_tgt = df['reason_start'].map(stats['reason_start_bucket']).fillna(gm)
    reason_combo = df['reason_start'].fillna('u') + '_' + df['reason_end'].fillna('u')
    reason_combo_tgt = reason_combo.map(stats['reason_combo_bucket']).fillna(gm)
    
    # Skip and shuffle patterns
    skip_tgt = df['skipped'].map(stats['skip_bucket']).fillna(gm)
    shuffle_tgt = df['shuffle'].map(stats['shuffle_bucket']).fillna(gm)
    
    # Basic features
    skipped = df['skipped'].fillna(0).astype(int)
    shuffle = df['shuffle'].fillna(0).astype(int)
    
    # Enhanced temporal features
    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    hour = ts.dt.hour.fillna(12)
    dow = ts.dt.dayofweek.fillna(3)
    month = ts.dt.month.fillna(6)
    day = ts.dt.day.fillna(15)
    
    # Cyclic encoding for hour and day of week
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    
    # Time period bins
    is_morning = ((hour >= 6) & (hour < 12)).astype(int)
    is_afternoon = ((hour >= 12) & (hour < 18)).astype(int)
    is_evening = ((hour >= 18) & (hour < 22)).astype(int)
    is_night = ((hour >= 22) | (hour < 6)).astype(int)
    
    # Build comprehensive feature matrix
    X = pd.DataFrame({
        # Smoothed target encodings (MOST IMPORTANT)
        'track_tgt': track_tgt,
        'track_std': track_std,
        'track_count_log': np.log1p(track_count),
        'track_freq_log': np.log1p(track_freq),
        'artist_tgt': artist_tgt,
        'artist_std': artist_std,
        'artist_freq_log': np.log1p(artist_freq),
        'album_tgt': album_tgt,
        'album_freq_log': np.log1p(album_freq),
        'platform_tgt': platform_tgt,
        'reason_end_tgt': reason_end_tgt,
        'reason_start_tgt': reason_start_tgt,
        'reason_combo_tgt': reason_combo_tgt,
        'skip_tgt': skip_tgt,
        'shuffle_tgt': shuffle_tgt,
        
        # Basic behavioral features
        'skipped': skipped,
        'shuffle': shuffle,
        
        # Enhanced temporal features
        'hour': hour,
        'dow': dow,
        'month': month,
        'day': day,
        'is_weekend': (dow >= 5).astype(int),
        'is_late_night': ((hour >= 22) | (hour <= 4)).astype(int),
        'is_morning': is_morning,
        'is_afternoon': is_afternoon,
        'is_evening': is_evening,
        'is_night': is_night,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        
        # Rich interaction features (critical for macro F1)
        'track_skip': track_tgt * skipped,
        'track_shuffle': track_tgt * shuffle,
        'track_skip_shuffle': track_tgt * skipped * shuffle,
        'reason_skip': reason_end_tgt * skipped,
        'reason_shuffle': reason_end_tgt * shuffle,
        'track_reason': track_tgt * reason_end_tgt,
        'track_reason_skip': track_tgt * reason_end_tgt * skipped,
        'artist_skip': artist_tgt * skipped,
        'artist_shuffle': artist_tgt * shuffle,
        'artist_platform': artist_tgt * platform_tgt,
        'platform_skip': platform_tgt * skipped,
        'platform_shuffle': platform_tgt * shuffle,
        'platform_hour': platform_tgt * hour,
        'track_platform': track_tgt * platform_tgt,
        'skip_shuffle': skipped * shuffle,
        'hour_skip': hour * skipped,
        'dow_skip': dow * skipped,
        'track_hour': track_tgt * hour,
        'artist_hour': artist_tgt * hour,
        'track_artist': track_tgt * artist_tgt,
        'track_album': track_tgt * album_tgt,
        'artist_album': artist_tgt * album_tgt,
        # Polynomial features for non-linear patterns
        'track_tgt_sq': track_tgt ** 2,
        'artist_tgt_sq': artist_tgt ** 2,
        'hour_sq': hour ** 2,
        # Ratio features
        'track_artist_ratio': track_tgt / (artist_tgt + 0.1),
        'track_platform_ratio': track_tgt / (platform_tgt + 0.1),
        # Combined temporal-behavioral
        'hour_weekend': hour * (dow >= 5).astype(int),
        'late_night_skip': ((hour >= 22) | (hour <= 4)).astype(int) * skipped,
        'weekend_shuffle': (dow >= 5).astype(int) * shuffle,
    })
    
    return X

X_train = make_features(train, stats)
feature_names = list(X_train.columns)
print(f"  Features: {len(feature_names)}", file=sys.stderr)
print(f"  Feature names: {feature_names}", file=sys.stderr)

print("\n[3/3] Training fast & effective ensemble for 0.55 Macro F1, 0.545 Accuracy, 60% classes F1>=0.40...", file=sys.stderr)

# Fast & effective: 2 strong models with optimized hyperparameters
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train.values, y, test_size=0.15, random_state=SEED, stratify=y)

models = []

if HAS_LGBM:
    # Model 1: Optimized for macro F1 - fast training with good performance
    print("  Training LightGBM model 1 (fast & effective)...", file=sys.stderr)
    model1 = LGBMClassifier(
        n_estimators=800,  # More estimators for better performance
        learning_rate=0.05,  # Balanced LR
        num_leaves=127,
        max_depth=15,
        min_child_samples=8,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.05,  # Lower regularization for better fit
        reg_lambda=0.1,
        random_state=SEED,
        objective='multiclass',
        num_class=50,
        n_jobs=-1,
        verbose=-1,
        class_weight='balanced',
        boosting_type='gbdt'
    )
    model1.fit(X_tr, y_tr, 
               eval_set=[(X_val, y_val)],
               callbacks=[lgbm.callback.early_stopping(stopping_rounds=60, verbose=False)])
    models.append(('lgbm1', model1, 0.6))
    
    # Model 2: Different config for diversity - deeper
    print("  Training LightGBM model 2 (diverse deep config)...", file=sys.stderr)
    model2 = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.04,
        num_leaves=150,
        max_depth=17,
        min_child_samples=6,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.03,
        reg_lambda=0.12,
        random_state=SEED + 1,
        objective='multiclass',
        num_class=50,
        n_jobs=-1,
        verbose=-1,
        class_weight='balanced',
        boosting_type='gbdt'
    )
    model2.fit(X_tr, y_tr,
               eval_set=[(X_val, y_val)],
               callbacks=[lgbm.callback.early_stopping(stopping_rounds=60, verbose=False)])
    models.append(('lgbm2', model2, 0.35))
    
    # Model 3: Conservative but stable for better class coverage
    print("  Training LightGBM model 3 (stable for class coverage)...", file=sys.stderr)
    model3 = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.06,
        num_leaves=90,
        max_depth=13,
        min_child_samples=12,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.08,
        reg_lambda=0.15,
        random_state=SEED + 2,
        objective='multiclass',
        num_class=50,
        n_jobs=-1,
        verbose=-1,
        class_weight='balanced',
        boosting_type='gbdt'
    )
    model3.fit(X_tr, y_tr,
               eval_set=[(X_val, y_val)],
               callbacks=[lgbm.callback.early_stopping(stopping_rounds=60, verbose=False)])
    models.append(('lgbm3', model3, 0.25))
    
    print(f"  ✓ 3 LightGBM models trained (fast & effective ensemble)", file=sys.stderr)
else:
    # Fallback: Single RandomForest (faster)
    from sklearn.ensemble import RandomForestClassifier
    print("  Training RandomForest (fast fallback)...", file=sys.stderr)
    
    model1 = RandomForestClassifier(
        n_estimators=300,  # Reduced for speed
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=SEED,
        n_jobs=-1
    )
    model1.fit(X_train.values, y)
    models.append(('rf1', model1, 1.0))
    
    print("  ✓ RandomForest model trained", file=sys.stderr)

model_path = OUTDIR / "model.pkl"
joblib.dump({'models': models, 'stats': stats, 'features': feature_names}, model_path)
print(f"\n✓ Ensemble ready with {len(models)} models", file=sys.stderr)
print(f"✓ Model saved to: {model_path}", file=sys.stderr)
print(f"✓ Model file exists: {model_path.exists()}", file=sys.stderr)
PYTHON_EOF

cat > /workdir/predict.py << 'PREDICT_SCRIPT'
#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

test = pd.read_csv(sys.argv[1])

# Load model - ensure path is correct
model_path = "/workdir/outputs/model.pkl"
if not Path(model_path).exists():
    print(f"ERROR: Model file not found at {model_path}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    if Path("/workdir/outputs").exists():
        print(f"Files in /workdir/outputs/: {list(Path('/workdir/outputs').iterdir())}", file=sys.stderr)
    else:
        print("Directory /workdir/outputs/ does not exist", file=sys.stderr)
    sys.exit(1)

print(f"Loading model from: {model_path}", file=sys.stderr)
data = joblib.load(model_path)
print("Model loaded successfully", file=sys.stderr)

models = data['models']
stats = data['stats']
features = data['features']

# IDENTICAL feature engineering (WITHOUT duration_ms) - Enhanced version
gm = stats['global_mean']

# Smoothed target encodings (use raw for high-count, smoothed for low-count)
track_count = test['track_uri'].map(stats['track_bucket_count']).fillna(0)
track_tgt_smooth = test['track_uri'].map(stats['track_bucket']).fillna(gm)
track_tgt_raw = test['track_uri'].map(stats['track_bucket_raw']).fillna(gm)
track_tgt = np.where(track_count >= 3, track_tgt_raw, track_tgt_smooth)
track_std = test['track_uri'].map(stats['track_bucket_std']).fillna(0)
track_freq = test['track_uri'].map(stats['track_freq']).fillna(0)

# Artist encoding
artist_count = test['artist'].map(stats.get('artist_bucket_count', {})).fillna(0)
artist_tgt_smooth = test['artist'].map(stats['artist_bucket']).fillna(gm)
artist_tgt_raw = test['artist'].map(stats['artist_bucket_raw']).fillna(gm)
artist_tgt = np.where(artist_count >= 3, artist_tgt_raw, artist_tgt_smooth)
artist_std = test['artist'].map(stats['artist_bucket_std']).fillna(0)
artist_freq = test['artist'].map(stats['artist_freq']).fillna(0)

# Album encoding
album_tgt = test['album'].map(stats['album_bucket']).fillna(gm)
album_freq = test['album'].map(stats['album_freq']).fillna(0)

# Platform patterns
platform_tgt = test['platform'].map(stats['platform_bucket']).fillna(gm)

# Reason patterns
reason_end_tgt = test['reason_end'].map(stats['reason_end_bucket']).fillna(gm)
reason_start_tgt = test['reason_start'].map(stats['reason_start_bucket']).fillna(gm)
reason_combo = test['reason_start'].fillna('u') + '_' + test['reason_end'].fillna('u')
reason_combo_tgt = reason_combo.map(stats['reason_combo_bucket']).fillna(gm)

# Skip and shuffle patterns
skip_tgt = test['skipped'].map(stats['skip_bucket']).fillna(gm)
shuffle_tgt = test['shuffle'].map(stats['shuffle_bucket']).fillna(gm)

# Basic features
skipped = test['skipped'].fillna(0).astype(int)
shuffle = test['shuffle'].fillna(0).astype(int)

# Enhanced temporal features
ts = pd.to_datetime(test['timestamp'], errors='coerce')
hour = ts.dt.hour.fillna(12)
dow = ts.dt.dayofweek.fillna(3)
month = ts.dt.month.fillna(6)
day = ts.dt.day.fillna(15)

# Cyclic encoding
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin = np.sin(2 * np.pi * dow / 7)
dow_cos = np.cos(2 * np.pi * dow / 7)

# Time period bins
is_morning = ((hour >= 6) & (hour < 12)).astype(int)
is_afternoon = ((hour >= 12) & (hour < 18)).astype(int)
is_evening = ((hour >= 18) & (hour < 22)).astype(int)
is_night = ((hour >= 22) | (hour < 6)).astype(int)

X = pd.DataFrame({
    'track_tgt': track_tgt,
    'track_std': track_std,
    'track_count_log': np.log1p(track_count),
    'track_freq_log': np.log1p(track_freq),
    'artist_tgt': artist_tgt,
    'artist_std': artist_std,
    'artist_freq_log': np.log1p(artist_freq),
    'album_tgt': album_tgt,
    'album_freq_log': np.log1p(album_freq),
    'platform_tgt': platform_tgt,
    'reason_end_tgt': reason_end_tgt,
    'reason_start_tgt': reason_start_tgt,
    'reason_combo_tgt': reason_combo_tgt,
    'skip_tgt': skip_tgt,
    'shuffle_tgt': shuffle_tgt,
    'skipped': skipped,
    'shuffle': shuffle,
    'hour': hour,
    'dow': dow,
    'month': month,
    'day': day,
    'is_weekend': (dow >= 5).astype(int),
    'is_late_night': ((hour >= 22) | (hour <= 4)).astype(int),
    'is_morning': is_morning,
    'is_afternoon': is_afternoon,
    'is_evening': is_evening,
    'is_night': is_night,
    'hour_sin': hour_sin,
    'hour_cos': hour_cos,
    'dow_sin': dow_sin,
    'dow_cos': dow_cos,
    'track_skip': track_tgt * skipped,
    'track_shuffle': track_tgt * shuffle,
    'track_skip_shuffle': track_tgt * skipped * shuffle,
    'reason_skip': reason_end_tgt * skipped,
    'reason_shuffle': reason_end_tgt * shuffle,
    'track_reason': track_tgt * reason_end_tgt,
    'track_reason_skip': track_tgt * reason_end_tgt * skipped,
    'artist_skip': artist_tgt * skipped,
    'artist_shuffle': artist_tgt * shuffle,
    'artist_platform': artist_tgt * platform_tgt,
    'platform_skip': platform_tgt * skipped,
    'platform_shuffle': platform_tgt * shuffle,
    'platform_hour': platform_tgt * hour,
    'track_platform': track_tgt * platform_tgt,
    'skip_shuffle': skipped * shuffle,
    'hour_skip': hour * skipped,
    'dow_skip': dow * skipped,
    'track_hour': track_tgt * hour,
    'artist_hour': artist_tgt * hour,
    'track_artist': track_tgt * artist_tgt,
    'track_album': track_tgt * album_tgt,
    'artist_album': artist_tgt * album_tgt,
    'track_tgt_sq': track_tgt ** 2,
    'artist_tgt_sq': artist_tgt ** 2,
    'hour_sq': hour ** 2,
    'track_artist_ratio': track_tgt / (artist_tgt + 0.1),
    'track_platform_ratio': track_tgt / (platform_tgt + 0.1),
    'hour_weekend': hour * (dow >= 5).astype(int),
    'late_night_skip': ((hour >= 22) | (hour <= 4)).astype(int) * skipped,
    'weekend_shuffle': (dow >= 5).astype(int) * shuffle,
})

# Ensure feature alignment - CRITICAL for LightGBM
print(f"Created {len(X.columns)} features in predict.py", file=sys.stderr)
print(f"Expected {len(features)} features from training", file=sys.stderr)

# Check for missing features
missing_features = set(features) - set(X.columns)
if missing_features:
    print(f"ERROR: Missing features: {missing_features}", file=sys.stderr)
    print(f"Available features: {sorted(X.columns)}", file=sys.stderr)
    sys.exit(1)

# Check for extra features
extra_features = set(X.columns) - set(features)
if extra_features:
    print(f"WARNING: Extra features will be removed: {extra_features}", file=sys.stderr)

# Select only the features used in training (in the EXACT order)
# This is critical - LightGBM requires exact feature order and count
X = X[features]

# Verify final feature count matches training
if len(X.columns) != len(features):
    print(f"ERROR: Feature count mismatch! Expected {len(features)}, got {len(X.columns)}", file=sys.stderr)
    sys.exit(1)

if list(X.columns) != features:
    print(f"ERROR: Feature order mismatch!", file=sys.stderr)
    print(f"Expected order: {features}", file=sys.stderr)
    print(f"Actual order: {list(X.columns)}", file=sys.stderr)
    # Reorder to match exactly
    X = X[features]

print(f"✓ Feature alignment verified: {len(X.columns)} features in correct order", file=sys.stderr)

# Ensemble prediction with weighted averaging of probabilities
all_probs = []
total_weight = 0

for name, model, weight in models:
    proba = model.predict_proba(X.values)
    all_probs.append(proba * weight)
    total_weight += weight

# Average probabilities and take argmax
avg_proba = sum(all_probs) / total_weight
preds = np.argmax(avg_proba, axis=1)

pd.DataFrame({
    'id': test['id'].astype(str),
    'prediction': preds.astype(int)
}).to_csv("/workdir/outputs/predictions.csv", index=False)

print("✓ Done")
PREDICT_SCRIPT

chmod +x /workdir/predict.py
echo "✓ Ready!"
