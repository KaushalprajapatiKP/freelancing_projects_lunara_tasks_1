# Task: Listening Duration Bucket Prediction (T-bench Format)

## Task Format: T-bench

This task follows the **T-bench** task format. See the main T-bench structure in task_0_t_bench_taxi_fare_predictor_v1 README.

## Problem Overview

**Task Type:** Multi-class Classification  
**Domain:** Music Streaming / User Engagement / Behavioral Analytics  
**Difficulty:** Hard  
**Category:** Machine Learning

Build a machine learning system to predict listening duration buckets for Spotify track plays using streaming metadata. Understanding user engagement with music streaming platforms is critical for recommendation systems, content curation, and user experience optimization.

## Task Configuration

From `task.yaml` metadata:
- **Difficulty:** hard
- **Category:** machine-learning
- **Tags:** tabular-data, music-streaming, user-engagement, multi-class-classification, classical-ml, time-series-features, behavioral-analytics
- **Time Limit:** 7200 seconds (2 hours)
- **Memory Limit:** 4096 MB
- **Max Agent Timeout:** 7200 seconds (2 hours)
- **Expert Time Estimate:** 180 minutes
- **Junior Time Estimate:** 360 minutes

## Problem Statement

Predict the listening duration bucket (0-49) for each Spotify track play as a multi-class classification task using all available metadata features **EXCEPT duration_ms itself**.

The duration buckets are created by:
- Taking actual listening duration (`duration_ms`)
- Discretizing into ~50 equal-frequency classes using quantile-based binning
- Creating balanced classes where each bucket represents ~2% of plays

## Data Structure

### Training Data
- **Location:** `/workdir/data/train.csv`
- **Format:** CSV with streaming metadata and `duration_bucket` target

### Test Data
- **Location:** `/tests/test.csv`
- **Format:** CSV with same schema but WITHOUT `duration_ms` and `duration_bucket` columns

### Output Format
- **File:** `/workdir/predict.py` (executable Python script)
- **Output File:** `/workdir/outputs/predictions.csv`
- **Required Columns:**
  - `id`: The play identifier (as string)
  - `prediction`: Duration bucket (integer from 0-49)

Example output:
```csv
id,prediction
1,23
2,45
3,12
```

## Data Schema

### Required Columns

- **id:** Unique play identifier (integer)
- **timestamp:** ISO 8601 timestamp of when the play occurred (string, e.g., "2014-07-13T23:56:17Z")
- **platform:** Device/platform used for playback (string, e.g., "Windows 7", "WebPlayer", "iOS")
- **duration_ms:** ACTUAL listening duration in milliseconds (integer, **AVAILABLE IN TRAINING ONLY - DO NOT USE AS FEATURE**)
- **track_name:** Name of the track played (string, may contain special characters)
- **artist:** Artist name (string, high cardinality)
- **album:** Album name (string, high cardinality)
- **track_uri:** Spotify URI for the track (string, format: "spotify:track:...")
- **reason_start:** Reason playback started (string, e.g., "playbtn", "trackdone", "fwdbtn", "popup", "unknown")
- **reason_end:** Reason playback ended (string, e.g., "trackdone", "playbtn", "fwdbtn", "unknown")
- **shuffle:** Whether shuffle mode was enabled (integer, 0 or 1)
- **skipped:** Whether the track was skipped (integer, 0 or 1)
- **source_file:** Original data file name (string, for reference)
- **duration_bucket:** TARGET VARIABLE (integer 0-49, **ONLY IN TRAINING DATA**)

## Critical Feature Engineering Requirements

To achieve the passing threshold, you MUST implement sophisticated feature engineering:

### 1. Temporal Features
- Extract hour of day (0-23) - users listen differently at different times
- Extract day of week (0-6) - weekday vs weekend patterns
- Create time period bins (morning, afternoon, evening, night)
- Cyclic encoding for hour/day (sin/cos transformations) to capture periodicity
- Extract month, day of month if useful for long-term patterns

### 2. Behavioral Features
- **reason_start/reason_end combinations:** (e.g., "playbtn→trackdone" indicates full listen)
- **Skip flag analysis:** skipped=1 correlates strongly with short durations
- **Shuffle mode impact:** shuffle=1 may correlate with lower engagement (background listening)
- Create interaction features between behavioral signals

### 3. Content-based Aggregations
- **Per-artist statistics:** mean/median/std of duration buckets, skip rates
- **Per-album statistics:** engagement metrics
- **Per-track statistics:** how this specific track is typically consumed
- Use target encoding or smoothed aggregations to handle high cardinality
- Handle cold-start: tracks/artists with few observations

### 4. Session-level Features
- Sequence patterns: is this play after skip, after full listen, etc.
- Time gaps between consecutive plays by same user (if timestamps suggest sessions)
- Platform switches or consistency

### 5. Platform Features
- Platform type encoding (mobile vs desktop vs web)
- Platform-specific listening patterns

### 6. Missing Value Handling
- Some categorical features may have missing or "unknown" values
- Create indicator features for missingness if informative

### 7. Advanced Encoding
- Use target encoding for high-cardinality features (artist, album, track_uri)
- Frequency encoding for categorical variables
- Consider smoothing techniques to prevent overfitting

## Solution Requirements

### 1. Training Script (`solution.sh`)

The solution must:
1. Create `/workdir/solution.sh` that implements the full ML pipeline
2. Load and validate the dataset schema
3. Analyze the data to understand streaming patterns
4. Engineer features from timestamps, categorical variables, and behavioral signals
5. Handle the multi-class classification problem with 50 balanced classes
6. Train one or more machine learning models optimized for Macro F1-Score
7. Tune hyperparameters to maximize cross-validation performance
8. Save model to `/workdir/outputs/model.pkl`
9. Create `/workdir/predict.py` that loads the trained model and generates predictions
10. Generate predictions for the test set and save to `/workdir/outputs/predictions.csv`

### 2. Prediction Script (`/workdir/predict.py`)

The prediction script must:
- Accept test CSV path as command-line argument: `python3 /workdir/predict.py <test_csv_path>`
- Load model from `/workdir/outputs/model.pkl`
- Apply the same feature engineering used during training
- Output predictions to `/workdir/outputs/predictions.csv`

## Evaluation

### Metric: Macro-averaged F1 Score (Macro F1)

**Formula:**
```
Macro F1 = (1/50) * Σ F1_i
```
Where F1_i is the F1 score for each of the 50 duration bucket classes.

### Passing Threshold

- **Passing Requirement:** Macro F1 score ≥ 0.555

This threshold ensures:
- Overall balanced performance across all classes
- Good performance across engagement patterns
- Prevents models from ignoring too many duration ranges

### Why This Threshold Is Challenging

- 50-class classification is inherently difficult (random baseline ~0.02 Macro F1)
- Macro F1 ≥ 0.555 requires good performance across engagement patterns
- Balanced classes mean you can't rely solely on majority class predictions
- Requires sophisticated feature engineering to capture listening patterns

## Available Packages

The environment includes:
- Python 3.x
- NumPy
- pandas
- scikit-learn
- scipy
- joblib
- lightgbm
- xgboost
- imbalanced-learn

**Disallowed:**
- Deep learning frameworks (TensorFlow, PyTorch, JAX)
- External API calls or internet access at runtime
- Using duration_ms as a feature (this is explicitly forbidden)
- Pre-trained embeddings or external data sources

## Environment Details

### Working Directory
- **Training/Development:** `/workdir`
- **Test Data:** `/tests`
- **Model Storage:** `/workdir/outputs/model.pkl`
- **Output:** `/workdir/outputs/predictions.csv`

## Technical Considerations

### Data Quality Notes

- duration_ms values range from 0 (instant skips) to several minutes
- The 50 duration buckets are equal-frequency, so each bucket has ~2% of training samples
- Some tracks appear multiple times with different listening durations (user behavior varies)
- Artist and album names are high-cardinality categorical variables (thousands of unique values)
- Timestamps span from 2014 to 2018 (multi-year history)
- reason_start/reason_end provide strong signals about engagement:
  - "trackdone" suggests track finished playing
  - "fwdbtn"/"backbtn" suggest skipping behavior
  - "playbtn" suggests manual selection
- skipped=1 is a strong indicator but doesn't tell the full story
- shuffle=1 may correlate with lower engagement (background listening)

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Handle mixed data types and high cardinality well
- **Multi-class classification:** Use appropriate objective functions
- **Ensemble Methods:** Combine multiple models for robustness
- **Hyperparameter Tuning:** Critical for 50-class classification

### Validation Strategy

- Use stratified k-fold cross-validation
- Monitor Macro F1 on validation sets
- Prevent overfitting through regularization
- Consider per-class performance to ensure balanced learning

## Success Criteria

To pass this task:
1. ✅ `/workdir/solution.sh` exists and is executable
2. ✅ `/workdir/predict.py` exists and is executable
3. ✅ Scripts run successfully
4. ✅ Predictions file has correct format (`id,prediction`)
5. ✅ All predictions are integers in [0, 49]
6. ✅ All test samples have predictions
7. ✅ **Macro F1 score ≥ 0.555**

## Additional Notes

- This task reflects real-world applications in music recommendation systems, personalized playlist generation, content discovery and curation, user engagement prediction, A/B testing for streaming platforms, and artist/label analytics
- Success on this task demonstrates ability to handle multi-class classification with many balanced classes, engineer features from temporal/behavioral/content metadata, work with high-cardinality categorical variables, optimize for macro metrics, and build production-grade ML pipelines for streaming analytics
- The balanced class distribution (each class ~2%) means you can't rely on majority class predictions
- Focus on capturing listening patterns through comprehensive feature engineering


