"""
grader.py — Spotify Listening Duration Bucket Prediction

Purpose
-------
This grader validates submissions for the Spotify Listening Duration Bucket Prediction task.
It implements a strict, sequential evaluation protocol: the first failed check causes an
immediate FAIL (score 0.0). Only when all checks pass does the grader compute the Macro F1
on the protected test set and return PASS (score 1.0) if the metric meets or exceeds the
configured threshold.

Security & Isolation
--------------------
- The grader assumes a secure environment where the agent cannot read `/tests/`.
- Ground-truth labels live in `/tests/test_ground_truth.csv` and are *never* exposed to the agent.
- The agent may only operate on `/workdir/data/` (training) and `/workdir` (its workspace).
- The grader expects the agent's solution to have already run and produced `/workdir/outputs/predictions.csv`.

High-level evaluation flow
--------------------------
1. Load ground truth from `/tests/test_ground_truth.csv`. Must contain header: id,target.
2. Verify that the participant provided `/workdir/solution.sh`.
3. Verify that `/workdir/predict.py` exists.
4. Execute `predict.py /tests/test.csv` to generate predictions.
5. Validate that `/workdir/outputs/predictions.csv` exists and has the exact header: id,prediction
   where prediction is an integer from 0-49 (duration bucket).
6. Check for extra prediction ids (predictions for ids not in ground truth).
7. Compute Macro-F1 between ground truth and predictions.
8. Compare Macro-F1 to PASS_THRESHOLD and return a binary GradingResult.

Return semantics
----------------
- Returns an instance of `apex_arena._types.GradingResult`.
- Final score is binary:
    * 1.0 => PASS (macro F1 >= PASS_THRESHOLD)
    * 0.0 => FAIL (any check failed OR macro F1 < PASS_THRESHOLD)
- The feedback string explains the failure or the metric summary.

Author: Kaushal Prajapati
"""

from __future__ import annotations

import csv
import os
import sys
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# The GradingResult dataclass (platform-provided) — expected return object
from apex_arena._types import GradingResult

# Metric computation
from sklearn.metrics import f1_score, classification_report, accuracy_score

# -----------------------------
# Configuration constants
# -----------------------------
# Location of the canonical (secure) ground truth file.
GROUND_TRUTH_PATH = Path("/tests/test_ground_truth.csv")

# Workspace where agent runs. The grader executes the agent's entrypoint inside /workdir.
WORKDIR = Path("/workdir")

# Agent's required entrypoint script. The grader will run this script (bash).
SOLUTION_SCRIPT = WORKDIR / "solution.sh"

# Expected location of predictions after agent runs.
PREDICTIONS_PATH = WORKDIR / "outputs" / "predictions.csv"

# Valid prediction range: duration buckets 0-49
MIN_BUCKET = 0
MAX_BUCKET = 49

# Execution timeout for the participant's solution.
# Must match max_agent_timeout_sec in task.yaml metadata
EXEC_TIMEOUT_SECONDS = 7200  # 2 hours

# Multi-threshold requirements to pass the task (must meet ALL)
# Primary: Overall balanced performance across all 50 classes
MACRO_F1_THRESHOLD = 0.55

# Tertiary: Ensure reasonable performance on individual classes
# At least 60% of classes should achieve F1 >= 0.40
MIN_CLASS_F1 = 0.40
MIN_CLASS_PERCENTAGE = 0.60  # 60% of classes

# -----------------------------
# Utility: file reading helpers
# -----------------------------
def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """
    Load the ground truth mapping from id -> duration_bucket from a secure CSV.

    Expected CSV header: id,target

    Returns:
      - (dict, None) on success where dict maps id -> bucket (int 0-49)
      - (None, error_message) on failure
    """
    if not path.exists():
        return None, f"Ground truth file not found at {path}"

    try:
        gt: Dict[str, int] = {}
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "id" not in reader.fieldnames or "target" not in reader.fieldnames:
                return None, "Ground truth CSV must include header columns: 'id' and 'target'."
            for row in reader:
                idx = row["id"]
                target_str = row["target"]
                try:
                    target = int(target_str)
                except ValueError:
                    return None, f"Ground truth contains non-integer target '{target_str}' for id '{idx}'."
                if not (MIN_BUCKET <= target <= MAX_BUCKET):
                    return None, f"Ground truth target {target} for id '{idx}' out of range [{MIN_BUCKET}, {MAX_BUCKET}]."
                gt[idx] = target
        if not gt:
            return None, "Ground truth file is empty."
        return gt, None
    except Exception as exc:
        return None, f"Failed to read ground truth CSV: {exc}"


def run_solution(script: Path = SOLUTION_SCRIPT, working_dir: Path = WORKDIR, timeout: int = EXEC_TIMEOUT_SECONDS) -> Tuple[bool, str, str]:
    """
    Execute the participant's solution script inside the working directory.

    Returns:
      - success_flag (True if returncode == 0)
      - stdout (captured)
      - stderr (captured)

    The grader purposely captures stdout/stderr to provide actionable feedback if execution fails.
    """
    if not script.exists():
        return False, "", f"Solution script not found at {script}"

    try:
        proc = subprocess.run(
            ["bash", str(script.name)],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (proc.returncode == 0, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout} seconds"
    except Exception as exc:
        return False, "", f"Error when executing solution script: {exc}"


def read_predictions(path: Path = PREDICTIONS_PATH) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """
    Read participant predictions from CSV and return a mapping id -> prediction.

    Expected CSV header EXACTLY: id,prediction (in that order).
    The function returns (preds_map, None) on success or (None, error_message) on failure.
    """
    if not path.exists():
        return None, f"Predictions file not found at {path}"

    try:
        preds: Dict[str, int] = {}
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            expected = ["id", "prediction"]
            if reader.fieldnames != expected:
                return None, f"Invalid CSV header. Expected: {expected}. Got: {reader.fieldnames}"
            row_no = 0
            for row in reader:
                row_no += 1
                idx = row["id"]
                pred_str = row["prediction"]
                if not idx:
                    return None, f"Empty id at CSV row {row_no}"
                try:
                    pred = int(pred_str)
                except ValueError:
                    return None, f"Invalid prediction '{pred_str}' at row {row_no}: must be integer"
                if not (MIN_BUCKET <= pred <= MAX_BUCKET):
                    return None, f"Prediction {pred} at row {row_no} out of range [{MIN_BUCKET}, {MAX_BUCKET}]"
                preds[idx] = pred
        if not preds:
            return None, "Predictions CSV is empty."
        return preds, None
    except Exception as exc:
        return None, f"Failed to read predictions CSV: {exc}"


# -----------------------------
# Primary grading function
# -----------------------------
def grade(_ctx=None) -> GradingResult:
    """
    Primary grader entry point.

    Sequential checks:
      0) Set permissions for test.csv and ensure outputs directory (grader runs as root)
      1) Load ground truth
      2) Check solution.sh exists
      3) Check predict.py exists
      4) Execute predict.py with test data
      5) Validate predictions file header & content
      6) Check for extra prediction ids
      7) Compute macro-F1
      8) Compare with PASS_THRESHOLD

    On first failure the function returns a GradingResult with score=0.0 and a helpful feedback message.
    On success, returns score=1.0 and feedback including the metric details.
    """
    # ---------------- Step 0: Ensure outputs directory exists ----------------
    try:
        outputs_dir = Path("/workdir/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(outputs_dir, 0o755)
        try:
            import pwd
            model_uid = pwd.getpwnam("model").pw_uid
            model_gid = pwd.getpwnam("model").pw_gid
            os.chown(outputs_dir, model_uid, model_gid)
        except:
            pass
    except:
        pass

    # ---------------- Step 1: Load ground truth ----------------
    ground_truth, error = load_ground_truth()
    if error:
        return GradingResult(
            score=0.0,
            subscores={"ground_truth_load": 0.0},
            weights={"ground_truth_load": 1.0},
            feedback=f"STEP 1 FAILED: {error}"
        )

    # ---------------- Step 2: Check solution.sh exists ----------------
    solution_script_path = SOLUTION_SCRIPT
    if not solution_script_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"solution_script_check": 0.0},
            weights={"solution_script_check": 1.0},
            feedback=f"STEP 2 FAILED: solution script not found at {SOLUTION_SCRIPT}\n\n"
                     "You must create a script at /workdir/solution.sh that:\n"
                     "- Trains your machine learning model\n"
                     "- Creates /workdir/predict.py for making predictions\n"
                     "- Saves your trained model for use by predict.py"
        )

    # ---------------- Step 3: Check if predict.py exists ----------------
    predict_script = Path("/workdir/predict.py")
    
    if not predict_script.exists():
        return GradingResult(
            score=0.0,
            subscores={"predict_script_check": 0.0},
            weights={"predict_script_check": 1.0},
            feedback="STEP 3 FAILED: predict.py not found at /workdir/predict.py\n\n"
                     "Your solution.sh must create a Python script that:\n"
                     "- Loads your trained model\n"
                     "- Accepts test CSV path as command-line argument\n"
                     "- Outputs predictions to /workdir/outputs/predictions.csv"
        )

    # ---------------- Step 4: Execute predict.py with test data ----------------
    test_csv_path = Path("/tests/test.csv")
    
    if not test_csv_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"test_data_check": 0.0},
            weights={"test_data_check": 1.0},
            feedback="STEP 4 FAILED: Test data not found at /tests/test.csv"
        )

    try:
        result = subprocess.run(
            [sys.executable, str(predict_script), str(test_csv_path)],
            cwd="/workdir",
            capture_output=True,
            text=True,
            timeout=7200
        )

        if result.returncode != 0:
            return GradingResult(
                score=0.0,
                subscores={"predict_execution": 0.0},
                weights={"predict_execution": 1.0},
                feedback=f"STEP 4 FAILED: predict.py execution failed\n\n"
                         f"Return code: {result.returncode}\n"
                         f"STDOUT:\n{result.stdout[:2000]}\n\n"
                         f"STDERR:\n{result.stderr[:2000]}\n\n"
                         f"Ensure your predict.py:\n"
                         f"- Loads the trained model successfully\n"
                         f"- Reads test data from sys.argv[1]\n"
                         f"- Handles all feature engineering\n"
                         f"- Outputs to /workdir/outputs/predictions.csv"
            )

    except subprocess.TimeoutExpired:
        return GradingResult(
            score=0.0,
            subscores={"predict_timeout": 0.0},
            weights={"predict_timeout": 1.0},
            feedback="STEP 4 FAILED: predict.py execution timed out (>7200 seconds)\n\n"
                     "Your prediction script is taking too long. Optimize your model loading and inference."
        )
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores={"predict_error": 0.0},
            weights={"predict_error": 1.0},
            feedback=f"STEP 4 FAILED: Error executing predict.py: {str(e)}"
        )

    # ---------------- Step 5: Read predictions ----------------
    predictions, error = read_predictions()
    if error:
        return GradingResult(
            score=0.0,
            subscores={"predictions_read": 0.0},
            weights={"predictions_read": 1.0},
            feedback=f"STEP 5 FAILED: {error}"
        )

    # ---------------- Step 6: Check for extra prediction ids ----------------
    gt_ids = set(ground_truth.keys())
    pred_ids = set(predictions.keys())
    
    extra = pred_ids - gt_ids
    
    # Only check for extra IDs, missing IDs are allowed
    if extra:
        sample = sorted(list(extra))[:10]
        return GradingResult(
            score=0.0,
            subscores={"extra_predictions": 0.0},
            weights={"extra_predictions": 1.0},
            feedback=f"STEP 6 FAILED: Found {len(extra)} extra predictions not in test set. Sample: {sample}"
        )

    # ---------------- Step 7: Compute metrics ----------------
    # Only compute metrics for IDs that exist in both ground truth and predictions
    common_ids = gt_ids & pred_ids
    if not common_ids:
        return GradingResult(
            score=0.0,
            subscores={"no_common_ids": 0.0},
            weights={"no_common_ids": 1.0},
            feedback="STEP 7 FAILED: No common IDs between ground truth and predictions"
        )
    
    # Filter to common IDs only
    filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in common_ids}
    filtered_predictions = {k: v for k, v in predictions.items() if k in common_ids}
    
    # Align predictions with ground truth (only common IDs)
    ids = sorted(common_ids)
    y_true = [filtered_ground_truth[i] for i in ids]
    y_pred = [filtered_predictions[i] for i in ids]
    
    # Compute overall metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Compute per-class F1 scores for detailed analysis
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    classes_above_threshold = sum(1 for f1 in per_class_f1 if f1 >= MIN_CLASS_F1)
    percentage_above_threshold = classes_above_threshold / len(per_class_f1)
    min_f1 = min(per_class_f1)
    max_f1 = max(per_class_f1)
    
    # ---------------- Step 8: Apply multi-threshold requirements ----------------
    # Both primary and tertiary thresholds must be met to pass
    meets_macro_f1 = macro_f1 >= MACRO_F1_THRESHOLD
    meets_class_distribution = percentage_above_threshold >= MIN_CLASS_PERCENTAGE
    
    if meets_macro_f1 and meets_class_distribution:
        feedback = textwrap.dedent(f"""
        ✓ PASSED! All thresholds met.
        
        PRIMARY THRESHOLD - Macro F1 Score:
        ✓ Achieved: {macro_f1:.4f} (required: ≥ {MACRO_F1_THRESHOLD:.2f})
        
        TERTIARY THRESHOLD - Class Distribution:
        ✓ Classes with F1 ≥ {MIN_CLASS_F1:.2f}: {classes_above_threshold}/50 ({percentage_above_threshold:.1%})
          (required: ≥ {MIN_CLASS_PERCENTAGE:.0%} of classes)
        
        Additional Metrics:
        - Accuracy:           {accuracy:.4f}
        - Micro F1:           {micro_f1:.4f}
        - Weighted F1:        {weighted_f1:.4f}
        - Min class F1:       {min_f1:.4f}
        - Max class F1:       {max_f1:.4f}
        
        Your model successfully predicts listening duration buckets across
        all 50 classes with excellent balanced performance. Outstanding work
        on sophisticated feature engineering and model optimization!
        """).strip()
        
        return GradingResult(
            score=1.0,
            subscores={
                "macro_f1_check": 1.0,
                "class_distribution_check": 1.0
            },
            weights={
                "macro_f1_check": 0.6,
                "class_distribution_check": 0.4
            },
            feedback=feedback
        )
    else:
        # Determine which thresholds failed
        failures = []
        if not meets_macro_f1:
            failures.append(f"Macro F1: {macro_f1:.4f} < {MACRO_F1_THRESHOLD:.2f} (gap: {MACRO_F1_THRESHOLD - macro_f1:.4f})")
        if not meets_class_distribution:
            failures.append(f"Class Coverage: {percentage_above_threshold:.1%} < {MIN_CLASS_PERCENTAGE:.0%} ({classes_above_threshold}/50 classes with F1 ≥ {MIN_CLASS_F1:.2f})")
        
        feedback = textwrap.dedent(f"""
        ✗ FAILED: One or more thresholds not met
        
        THRESHOLD STATUS:
        {'✓' if meets_macro_f1 else '✗'} Macro F1:        {macro_f1:.4f} (required: ≥ {MACRO_F1_THRESHOLD:.2f})
        {'✓' if meets_class_distribution else '✗'} Class Coverage:  {classes_above_threshold}/50 ({percentage_above_threshold:.1%}) (required: ≥ {MIN_CLASS_PERCENTAGE:.0%} with F1 ≥ {MIN_CLASS_F1:.2f})
        
        FAILED REQUIREMENTS:
        {chr(10).join(f'  - {f}' for f in failures)}
        
        Additional Metrics:
        - Accuracy:           {accuracy:.4f}
        - Micro F1:           {micro_f1:.4f}
        - Weighted F1:         {weighted_f1:.4f}
        - Min class F1:        {min_f1:.4f}
        - Max class F1:        {max_f1:.4f}
        
        To meet ALL thresholds:
        
        1. **Advanced Feature Engineering (Critical for {MACRO_F1_THRESHOLD:.0%} Macro F1):**
           - Rich temporal features: hour, day, cyclic encoding (sin/cos), time periods
           - Target encoding for track_uri/artist/album with proper train statistics
           - Behavioral combinations: reason_start+end patterns, skip interactions
           - Platform and shuffle mode patterns
           - Interaction features between temporal, behavioral, and content signals
        
        2. **High-Cardinality Categorical Handling:**
           - Use smoothed target encoding to prevent overfitting
           - Track/artist/album historical bucket patterns (NOT duration_ms!)
           - Frequency-based encodings for rare entities
           - Handle cold-start tracks/artists gracefully
        
        3. **Advanced Model Optimization:**
           - Use LightGBM or XGBoost with 50-class multiclass objective
           - Tune for Macro F1 specifically (not accuracy or logloss)
           - Consider ensemble of multiple models with different hyperparameters
           - Use large n_estimators (800-1200) with early stopping
           - Carefully tune max_depth, learning_rate, subsample parameters
        
        4. **Ensure Balanced Class Performance:**
           - Macro F1 = {MACRO_F1_THRESHOLD:.0%} requires ALL 50 classes perform well
           - Analyze per-class F1 scores to find weak classes
           - Ensure model doesn't over-predict common duration ranges
           - {MIN_CLASS_PERCENTAGE:.0%} of classes must achieve F1 ≥ {MIN_CLASS_F1:.2f}
        """).strip()
        
        return GradingResult(
            score=0.0,
            subscores={
                "macro_f1_check": 1.0 if meets_macro_f1 else 0.0,
                "class_distribution_check": 1.0 if meets_class_distribution else 0.0
            },
            weights={
                "macro_f1_check": 0.6,
                "class_distribution_check": 0.4
            },
            feedback=feedback
        )


if __name__ == "__main__":
    result = grade()
    print(f"\nFinal Score: {result.score}")
    print(f"Feedback:\n{result.feedback}")
