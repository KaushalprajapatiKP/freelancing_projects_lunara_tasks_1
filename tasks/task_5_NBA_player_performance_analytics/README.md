# Task: NBA Player Performance Analytics (T-bench Format)

## Task Format: T-bench

This task follows the **T-bench** task format. See the main T-bench structure in task_0_t_bench_taxi_fare_predictor_v1 README.

## Problem Overview

**Task Type:** Multi-class Classification  
**Domain:** Sports Analytics / Basketball  
**Difficulty:** Hard  
**Category:** Machine Learning

Build a machine learning system to predict NBA player performance buckets using game and player metadata. The target variable is created by discretizing performance metrics into ~50 equal-frequency classes, creating a balanced multi-class classification problem.

## Task Configuration

From `task.yaml` metadata:
- **Difficulty:** hard
- **Category:** machine-learning
- **Tags:** tabular-data, sports-analytics, player-performance, multi-class-classification, classical-ml, basketball-analytics
- **Time Limit:** 7200 seconds (2 hours)
- **Memory Limit:** 4096 MB
- **Max Agent Timeout:** 7200 seconds (2 hours)
- **Expert Time Estimate:** 180 minutes
- **Junior Time Estimate:** 360 minutes

## Problem Statement

Predict the performance bucket (0-49) for each NBA player game as a multi-class classification task using all available game and player metadata **EXCEPT the direct performance metric itself**.

The performance buckets are created by:
- Taking actual performance metric (e.g., minutes_played, game_score, or PER)
- Discretizing into ~50 equal-frequency classes using quantile-based binning
- Creating balanced classes where each bucket represents ~2% of games

## Data Structure

### Training Data
- **Location:** `/workdir/data/train.csv`
- **Format:** CSV with game and player metadata and `performance_bucket` target

### Test Data
- **Location:** `/tests/test.csv`
- **Format:** CSV with same schema but WITHOUT `performance_metric` and `performance_bucket` columns

### Output Format
- **File:** `/workdir/predict.py` (executable Python script)
- **Output File:** `/workdir/outputs/predictions.csv`
- **Required Columns:**
  - `id`: The game identifier (as string)
  - `prediction`: Performance bucket (integer from 0-49)

Example output:
```csv
id,prediction
1,23
2,45
3,12
```

## Data Schema

### Required Columns

- **id:** Unique game identifier (integer)
- **player_name:** Name of the player (string)
- **team_abbreviation:** Player's team code (string, e.g., "LAL", "GSW")
- **age:** Player's age at game time (float)
- **player_height:** Height in cm (float)
- **player_weight:** Weight in kg (float)
- **college:** College attended (string, may be null)
- **country:** Country of origin (string)
- **draft_year:** Year drafted (string or integer, may be "Undrafted")
- **draft_round:** Draft round (string or integer, may be "Undrafted")
- **draft_number:** Draft pick number (string or integer, may be "Undrafted")
- **season:** NBA season (string, e.g., "2015-16")
- **pts:** Points scored (integer)
- **reb:** Rebounds (integer)
- **ast:** Assists (integer)
- **net_rating:** Team net rating (float)
- **oreb_pct:** Offensive rebound percentage (float)
- **dreb_pct:** Defensive rebound percentage (float)
- **usg_pct:** Usage percentage (float)
- **ts_pct:** True shooting percentage (float)
- **ast_pct:** Assist percentage (float)
- **performance_metric:** Actual performance value (float, **ONLY IN TRAINING DATA - DO NOT USE AS FEATURE**)
- **performance_bucket:** Target variable (integer 0-49, **ONLY IN TRAINING DATA**)

### Data Limitations

- The dataset does NOT include: game dates, opponent identifiers, home/away indicators, or game ordering
- Focus on features that can be derived from the available columns listed above

## Solution Requirements

### 1. Training Script (`solution.sh`)

The solution must:
1. Create `/workdir/solution.sh` that implements the full ML pipeline
2. Load and validate the dataset schema
3. Analyze the data to understand performance patterns
4. Engineer features from available columns
5. Handle the multi-class classification problem with 50 balanced classes
6. Train one or more machine learning models optimized for Macro F1-Score
7. Save the trained model(s) and preprocessing artifacts to `/workdir/outputs/model.pkl` (or `/workdir/model.pkl`)
8. Create `/workdir/predict.py` for inference
9. Tune hyperparameters to maximize cross-validation performance
10. Generate predictions for the test set and save to `/workdir/outputs/predictions.csv`

### 2. Prediction Script (`/workdir/predict.py`)

The prediction script must:
- Accept two command-line arguments: `model_path` (first arg) and `test_csv_path` (second arg)
- Load the saved model from `model_path`
- Read test data from `test_csv_path`
- Apply the same feature engineering used during training
- Generate predictions and save to `/workdir/outputs/predictions.csv`

## Feature Engineering Suggestions

### Player Characteristics
- Use player_name, team_abbreviation, college, country as categorical features
- Encode high-cardinality categoricals (player_name, college) using target encoding or frequency encoding
- Create player-level aggregations (historical performance stats per player)

### Physical Attributes
- Use age, player_height, player_weight directly
- Create derived features: BMI, height/weight ratio
- Consider age groups or bins

### Draft Information
- Parse draft_year, draft_round, draft_number
- Create "Undrafted" indicator
- Create draft position features (lower is better)
- Consider draft year as experience indicator

### Game Statistics
- Use pts, reb, ast directly
- Create derived features:
  - Total contributions: pts + reb + ast
  - Efficiency metrics: pts per minute (if available)
  - Triple-double indicator (pts ≥ 10 and reb ≥ 10 and ast ≥ 10)

### Advanced Statistics
- Use net_rating, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct
- Create interaction features:
  - usg_pct × ts_pct (usage efficiency)
  - net_rating × usg_pct (team impact)
- Normalize or scale these features

### Season Features
- Parse season string (e.g., "2015-16")
- Extract year or season number
- Create season progression features if multiple seasons present

### Interaction Features
- Player × Team interactions
- Draft position × Age interactions
- Statistics × Physical attributes

## Evaluation

### Metric: Macro-averaged F1 Score (Macro F1)

**Formula:**
```
Macro F1 = (1/50) * Σ F1_i
```
Where F1_i is the F1 score for each of the 50 performance bucket classes.

### Performance Expectations

This is a challenging multi-class classification problem with 50 balanced classes (each class ~2% of data). The balanced class distribution means each bucket represents approximately 2% of the data, requiring models that can learn subtle patterns across all performance levels without overfitting to specific classes.

Achieving strong performance requires:
- Comprehensive feature engineering leveraging all available metadata
- Advanced ensemble methods (LightGBM, XGBoost with proper hyperparameter tuning)
- Careful handling of high-cardinality categorical variables (player names, colleges, teams)
- Robust cross-validation strategies to ensure generalization
- Balanced performance across all 50 classes (avoiding models that excel on some classes while failing on others)

Strong models demonstrate:
- High overall Macro F1 score indicating good average performance
- Consistent per-class F1 scores with no catastrophic failures on individual classes
- Good coverage across the full range of performance buckets

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
- Using performance_metric as a feature (explicitly forbidden)
- Pre-trained embeddings or external data sources
- Reading or accessing ground truth labels during training or inference

## Environment Details

### Working Directory
- **Training/Development:** `/workdir`
- **Test Data:** `/tests`
- **Model Storage:** `/workdir/outputs/model.pkl` or `/workdir/model.pkl`
- **Output:** `/workdir/outputs/predictions.csv`

## Technical Considerations

### Model Selection

Recommended approaches:
- **LightGBM/XGBoost:** Handle mixed data types and high cardinality well
- **Multi-class classification:** Use appropriate objective functions
- **Ensemble Methods:** Combine multiple models for robustness
- **Hyperparameter Tuning:** Critical for 50-class classification

### Handling High Cardinality

- **player_name:** Thousands of unique players - use target encoding or frequency encoding
- **college:** Many unique colleges - similar encoding approach
- **team_abbreviation:** Limited set - can use one-hot or label encoding

### Validation Strategy

- Use stratified k-fold cross-validation
- Monitor Macro F1 on validation sets
- Prevent overfitting through regularization
- Consider per-class performance to ensure balanced learning

### Data Context

This task uses real NBA player game data with mixed numeric and categorical features. The dataset includes high-cardinality categorical variables (players, teams, colleges).

## Success Criteria

To pass this task:
1. ✅ `/workdir/solution.sh` exists and is executable
2. ✅ `/workdir/predict.py` exists and is executable
3. ✅ Scripts run successfully
4. ✅ Predictions file has correct format (`id,prediction`)
5. ✅ All predictions are integers in [0, 49]
6. ✅ All test samples have predictions
7. ✅ Meets evaluation threshold (see task.yaml for specific threshold)

## Additional Notes

- The balanced class distribution (each class ~2%) means you can't rely on majority class predictions
- Focus on comprehensive feature engineering from all available metadata
- Consider player-level historical aggregations if the same players appear multiple times
- The lack of opponent/home-away information means you must work with available features only


