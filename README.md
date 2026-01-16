# Lunara: Machine Learning Task Collection

A comprehensive collection of machine learning tasks for evaluation and benchmarking, supporting both **Harbor** and **T-bench** task formats. This repository contains 12 diverse machine learning challenges covering classification, regression, and multi-label problems across various domains.

## 📋 Table of Contents

- [Overview](#overview)
- [Task Formats](#task-formats)
  - [Harbor Format](#harbor-format)
  - [T-bench Format](#t-bench-format)
- [Task List](#task-list)
  - [Harbor Format Tasks](#harbor-format-tasks)
  - [T-bench Format Tasks](#t-bench-format-tasks)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Task Details](#task-details)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Overview

Lunara is a curated collection of machine learning tasks designed for:
- **Model Evaluation:** Test and benchmark ML models across diverse problem types
- **Skill Assessment:** Evaluate machine learning engineering capabilities
- **Research & Development:** Provide standardized tasks for ML research
- **Education:** Offer practical ML challenges for learning

Each task includes:
- Complete dataset (training and test splits)
- Detailed instructions and requirements
- Evaluation scripts and metrics
- Docker environment configuration
- Comprehensive documentation

## Task Formats

This repository supports two task formats: **Harbor** and **T-bench**. Each format has its own structure and conventions.

### Harbor Format

Harbor tasks use a standardized structure with TOML configuration and separate instruction files.

#### Harbor Task Structure

```
task_name/
├── task.toml                    # Task configuration (metadata, timeouts, resources)
├── instruction.md               # Detailed task instructions
├── environment/
│   ├── Dockerfile              # Container environment definition
│   └── data/
│       └── train.csv           # Training data
├── solution/
│   └── solve.sh                # Solution script entry point
└── tests/
    ├── test.sh                 # Test execution script
    ├── test_outputs.py         # Test validation logic
    ├── test.csv                # Test data (without labels)
    └── test_ground_truth.csv   # Ground truth for evaluation
```

#### Harbor Configuration (`task.toml`)

```toml
version = "1.0"

[metadata]
difficulty = "medium"
category = "MLE"
tags = ["regression", "machine-learning", "tabular-data"]

[verifier]
timeout_sec = 3600.0

[agent]
timeout_sec = 3600.0

[environment]
build_timeout_sec = 600.0
memory_mb = 2048
```

#### Harbor Key Features

- **Working Directory:** `/app`
- **Model Storage:** `/app/model.pkl`
- **Output:** `/app/predictions.csv`
- **Test Execution:** Via `tests/test.sh` using pytest
- **Solution Entry:** `solution/solve.sh`

### T-bench Format

T-bench tasks use YAML configuration and integrate grading directly into the task structure.

#### T-bench Task Structure

```
task_name/
├── task.yaml                    # Task configuration and prompt
├── Dockerfile                   # Container environment
├── grader.py                    # Grading/evaluation script
├── solution.sh                  # Solution script entry point
├── data/
│   └── train.csv               # Training data
└── tests/
    ├── test.csv                # Test data (without labels)
    └── test_ground_truth.csv   # Ground truth for evaluation
```

#### T-bench Configuration (`task.yaml`)

```yaml
prompt: |
  Task description and instructions...
  
metadata:
  difficulty: medium
  category: MLE
  tags:
    - regression
    - machine-learning
  time_limit: 300
  memory_limit: 512
  max_agent_timeout_sec: 600
  expert_time_estimate_min: 90
  junior_time_estimate_min: 180
```

#### T-bench Key Features

- **Working Directory:** `/workdir`
- **Model Storage:** `/workdir/model.pkl` or `/workdir/outputs/model.pkl`
- **Output:** `/workdir/predictions.csv` or `/workdir/outputs/predictions.csv`
- **Test Execution:** Via `grader.py` script
- **Solution Entry:** `solution.sh`

## Task List

### Harbor Format Tasks

| Task ID | Task Name | Type | Difficulty | Domain |
|---------|-----------|------|------------|--------|
| [task_0_harbor](tasks/task_0_harbor-predicting-euphoria-in-the-streets/README.md) | Predicting Euphoria in the Streets | Binary Classification | Hard | Healthcare |
| [task_6](tasks/task_6_country_level_inflation/README.md) | Country-Level Inflation Prediction | Regression | Medium | Economics |
| [task_7](tasks/task_7_spring_classification_competition/README.md) | Spring Classification Competition | Binary Classification | Medium | Healthcare |
| [task_8](tasks/task_8_nifty_stock_movement/README.md) | NIFTY Stock Movement Prediction | Multi-class Classification | Medium | Finance/NLP |
| [task_9](tasks/task_9_regression_yield_prediction/README.md) | Regression Yield Prediction | Regression | Medium | Agriculture |
| [task_10](tasks/task_10_thapar_summer_school_hack_i/README.md) | Thapar Summer School Hack-I | Regression | Medium | General |
| [task_11](tasks/task_11_airbnb_price_prediction/README.md) | Airbnb Price Prediction | Regression | Medium | Real Estate |

### T-bench Format Tasks

| Task ID | Task Name | Type | Difficulty | Domain |
|---------|-----------|------|------------|--------|
| [task_0_t_bench](tasks/task_0_t_bench_taxi_fare_predictor_v1/README.md) | Taxi Fare Predictor v1 | Regression | Medium | Transportation |
| [task_1](tasks/task_1_steel_plate_defect_prediction/README.md) | Steel Plate Defect Prediction | Multi-label Classification | Very Hard | Manufacturing |
| [task_2](tasks/task_2_global_ev_charging_stations/README.md) | Global EV Charging Stations | Binary Classification | Hard | EV Infrastructure |
| [task_3](tasks/task_3_academic_risk_prediction/README.md) | Academic Risk Prediction | Classification | Medium | Education |
| [task_4](tasks/task_4_listening_duration_bucket_prediction_from_user_streaming_metadata/README.md) | Listening Duration Bucket Prediction | Multi-class Classification | Hard | Music Streaming |
| [task_5](tasks/task_5_NBA_player_performance_analytics/README.md) | NBA Player Performance Analytics | Multi-class Classification | Hard | Sports Analytics |

## Project Structure

```
Lunara/
├── README.md                    # This file
├── Setup/                       # Setup and installation scripts
│   ├── README.md               # Setup instructions
│   ├── installation.sh         # Installation script
│   └── TASK_REVIEWING_SKILL.md # Task creation guidelines
├── tasks/                      # All task directories
│   ├── task_0_harbor-*/        # Harbor format tasks
│   ├── task_0_t_bench_*/        # T-bench format tasks
│   ├── task_1_*/               # Individual tasks
│   └── ...
└── apex_env/                   # Python virtual environment
```

## Getting Started

### Prerequisites

- Python 3.12+
- Docker (for containerized execution)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Lunara
   ```

2. **Set up the virtual environment:**
   ```bash
   source apex_env/bin/activate  # On macOS/Linux
   # or
   apex_env\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt  # If available
   ```

4. **Verify installation:**
   ```bash
   apex-arena --version
   ```

### Running a Task

#### For Harbor Format Tasks

1. Navigate to the task directory:
   ```bash
   cd tasks/task_0_harbor-predicting-euphoria-in-the-streets
   ```

2. Review the task documentation:
   ```bash
   cat README.md
   cat instruction.md
   ```

3. Build the Docker environment:
   ```bash
   docker build -t task-env -f environment/Dockerfile .
   ```

4. Run the solution:
   ```bash
   docker run -v $(pwd):/app task-env bash solution/solve.sh
   ```

5. Test the solution:
   ```bash
   docker run -v $(pwd):/app task-env bash tests/test.sh
   ```

#### For T-bench Format Tasks

1. Navigate to the task directory:
   ```bash
   cd tasks/task_0_t_bench_taxi_fare_predictor_v1
   ```

2. Review the task documentation:
   ```bash
   cat README.md
   cat task.yaml
   ```

3. Test the solution:
   ```bash
   apex-arena test-solution task_0_t_bench_taxi_fare_predictor_v1
   ```

## Task Details

Each task includes comprehensive documentation covering:

- **Problem Overview:** Task type, domain, difficulty level
- **Data Structure:** Training/test data formats and locations
- **Features:** Detailed feature descriptions
- **Solution Requirements:** Training and prediction script specifications
- **Evaluation:** Metrics, formulas, and passing thresholds
- **Technical Considerations:** Preprocessing, model selection, validation strategies
- **Success Criteria:** Checklist for passing the task

### Quick Reference: Task Metrics

| Task | Metric | Threshold |
|------|--------|-----------|
| task_0_harbor | AUC-ROC | ≥ 0.787 |
| task_0_t_bench | MAE | ≤ $3.64 (Score ≥ 96%) |
| task_1 | Mean ROC-AUC | ≥ 0.8835 |
| task_2 | Macro F1 | ≥ 0.5 |
| task_3 | (See task README) | - |
| task_4 | Macro F1 | ≥ 0.555 |
| task_5 | Macro F1 | (See task README) |
| task_6 | MAE, RMSE, R² | MAE ≤ 6.0, RMSE ≤ 11.0, R² ≥ 0.4 |
| task_7 | F1-Score | ≥ 0.305 |
| task_8 | Accuracy | ≥ 45.3% |
| task_9 | MAE | ≤ 250.0 |
| task_10 | R² | ≥ 0.38 |
| task_11 | RMSE | ≤ 111.0 |

## Evaluation

### Evaluation Process

1. **Solution Execution:** The solution script trains a model and creates a prediction script
2. **Prediction Generation:** The prediction script generates predictions on test data
3. **Format Validation:** Output format is validated (columns, data types, completeness)
4. **Metric Calculation:** Evaluation metrics are computed
5. **Threshold Check:** Results are compared against passing thresholds

### Common Evaluation Metrics

- **Classification:**
  - Accuracy: Proportion of correct predictions
  - F1-Score: Harmonic mean of precision and recall
  - Macro F1: Average F1 across all classes
  - AUC-ROC: Area under the ROC curve

- **Regression:**
  - MAE: Mean Absolute Error
  - RMSE: Root Mean Squared Error
  - R²: Coefficient of Determination

### Binary Scoring

Most tasks use binary scoring:
- **Pass:** Score = 1.0 (meets or exceeds threshold)
- **Fail:** Score = 0.0 (below threshold)

Some tasks use continuous scoring with linear scaling between thresholds.

## Available Packages

### Common Packages (Most Tasks)

- Python 3.x (standard library)
- NumPy
- pandas
- scikit-learn
- joblib

### Additional Packages (Task-Specific)

- **LightGBM:** Available in most tasks
- **XGBoost:** Available in most tasks
- **scipy:** Available in most tasks
- **imbalanced-learn:** Available in some classification tasks

### Restrictions

- **Deep Learning:** TensorFlow, PyTorch, JAX are generally NOT available
- **External APIs:** No internet access at runtime
- **Pre-trained Models:** Generally not allowed

## Contributing

### Creating New Tasks

See `Setup/README.md` and `Setup/TASK_REVIEWING_SKILL.md` for detailed guidelines on creating new tasks.

### Task Creation Workflow

1. Choose a competition/dataset
2. Download and prepare data
3. Create task structure (Harbor or T-bench format)
4. Implement solution script
5. Create grader/evaluation script
6. Test locally
7. Push to Apex Arena
8. Run evaluations and adjust thresholds

### Guidelines

- Follow the existing task format (Harbor or T-bench)
- Include comprehensive README.md documentation
- Set appropriate difficulty levels
- Ensure passing thresholds are challenging but achievable
- Test thoroughly before submission

## Task-Specific Documentation

Each task has its own detailed README.md file. Navigate to individual task directories for:

- Complete feature descriptions
- Detailed evaluation criteria
- Technical implementation guidance
- Domain-specific considerations
- Success criteria checklists

## Resources

- [Apex Arena Documentation](https://apex-arena.readthedocs.io/)
- [Harbor Task Format](https://github.com/harbor-ml/harbor)
- [T-bench Documentation](https://github.com/t-bench/t-bench)

## License

[Add license information here]

## Contact

[Add contact information here]

---

**Note:** This repository contains machine learning tasks for evaluation purposes. Each task includes datasets, evaluation scripts, and documentation. For detailed information about a specific task, refer to its individual README.md file in the `tasks/` directory.

