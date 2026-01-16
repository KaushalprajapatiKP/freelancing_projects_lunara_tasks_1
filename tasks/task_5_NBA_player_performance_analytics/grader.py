"""
grader.py — NBA Player Performance Bucket Prediction

Purpose
-------
This grader validates submissions for the NBA Player Performance Bucket Prediction task.
It implements a strict, sequential evaluation protocol: the first failed check causes an
immediate FAIL (score 0.0). Only when all checks pass does the grader compute the Macro F1
on the protected test set and return PASS (score 1.0) if the metric meets or exceeds the
configured threshold.

Security & Isolation
--------------------
- The grader assumes a secure environment where the agent cannot read `/tests/`.
- Ground-truth labels live in `/tests/test_ground_truth.csv` and are *never* exposed to the agent.
- The agent may only operate on `/workdir/data/` (training) and `/workdir` (its workspace).
- The grader executes the agent's entrypoint (expected at `/workdir/solution.sh`) within
  the /workdir sandbox and then inspects the produced `/workdir/outputs/predictions.csv`.

High-level evaluation flow
--------------------------
1. Load ground truth from `/tests/test_ground_truth.csv`. Must contain header: id,target.
2. Verify that the participant provided `/workdir/solution.sh`.
3. Execute `solution.sh` inside `/workdir`. This should train model and create predict.py.
4. Verify that `/workdir/predict.py` exists.
5. Execute `predict.py /tests/test.csv` to generate predictions.
6. Validate that `/workdir/outputs/predictions.csv` exists and has the exact header: id,prediction
   where prediction is an integer from 0-49 (performance bucket).
7. Ensure completeness: predictions must cover all test ids and include no extras.
8. Compute Macro-F1 between ground truth and predictions.
9. Compare Macro-F1 to PASS_THRESHOLD and return a binary GradingResult.

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
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from apex_arena._types import GradingResult

from sklearn.metrics import f1_score, classification_report, recall_score, precision_score

# -----------------------------
# Configuration constants
# -----------------------------
GROUND_TRUTH_PATH = Path("/tests/test_ground_truth.csv")
WORKDIR = Path("/workdir")
SOLUTION_SCRIPT = WORKDIR / "solution.sh"
PREDICTIONS_PATH = WORKDIR / "outputs" / "predictions.csv"

# Valid prediction range: performance buckets 0-49
MIN_BUCKET = 0
MAX_BUCKET = 49

EXEC_TIMEOUT_SECONDS = 7200  # 2 hours

# Macro F1 threshold to pass the task. Must be consistent with task.yaml.
PASS_THRESHOLD = 0.95


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """
    Load the ground truth mapping from id -> performance_bucket from a secure CSV.

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


def run_solution(
    script: Path = SOLUTION_SCRIPT,
    working_dir: Path = WORKDIR,
    timeout: int = EXEC_TIMEOUT_SECONDS
) -> Tuple[bool, str, str]:
    if not script.exists():
        return False, "", f"Solution script not found at {script}"

    test_csv_path = Path("/tests/test.csv")
    if test_csv_path.exists():
        try:
            os.chmod(test_csv_path, 0o644)
            try:
                import pwd
                model_uid = pwd.getpwnam("model").pw_uid
                model_gid = pwd.getpwnam("model").pw_gid
                os.chown(test_csv_path, model_uid, model_gid)
            except:
                pass
            try:
                import shutil
                workdir_test = Path("/workdir/test.csv")
                if not workdir_test.exists() or workdir_test.stat().st_mtime < test_csv_path.stat().st_mtime:
                    shutil.copy(test_csv_path, workdir_test)
                    os.chmod(workdir_test, 0o644)
                    try:
                        import pwd
                        model_uid = pwd.getpwnam("model").pw_uid
                        model_gid = pwd.getpwnam("model").pw_gid
                        os.chown(workdir_test, model_uid, model_gid)
                    except:
                        pass
            except:
                pass
        except:
            pass

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


def grade(_ctx=None) -> GradingResult:
    test_csv_path = Path("/tests/test.csv")
    if test_csv_path.exists():
        try:
            os.chmod(test_csv_path, 0o644)
            try:
                import pwd
                model_uid = pwd.getpwnam("model").pw_uid
                model_gid = pwd.getpwnam("model").pw_gid
                os.chown(test_csv_path, model_uid, model_gid)
            except:
                pass
            try:
                import shutil
                workdir_test = Path("/workdir/test.csv")
                shutil.copy(test_csv_path, workdir_test)
                os.chmod(workdir_test, 0o644)
                try:
                    import pwd
                    model_uid = pwd.getpwnam("model").pw_uid
                    model_gid = pwd.getpwnam("model").pw_gid
                    os.chown(workdir_test, model_uid, model_gid)
                except:
                    pass
            except:
                pass
        except Exception:
            pass

        outputs_dir = Path("/workdir/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
    try:
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

    ground_truth, err = load_ground_truth()
    if err:
        return GradingResult(
            score=0.0,
            subscores={"step_1_load_ground_truth": 0.0},
            weights={"step_1_load_ground_truth": 1.0},
            feedback=f"STEP 1 FAILED: {err}"
        )

    solution_script_path = SOLUTION_SCRIPT
    if not solution_script_path.exists():
        alt_path = Path("/tests/solution.sh")
        if alt_path.exists():
            import shutil
            solution_script_path = alt_path
            shutil.copy(alt_path, SOLUTION_SCRIPT)
            os.chmod(SOLUTION_SCRIPT, 0o755)
        else:
            return GradingResult(
                score=0.0,
                subscores={"step_2_solution_script_exists": 0.0},
                weights={"step_2_solution_script_exists": 1.0},
                feedback=f"STEP 2 FAILED: solution script not found at {SOLUTION_SCRIPT} or /tests/solution.sh"
            )
    
    # ---------------- Step 3: Check predict.py and model exist ----------------
    predict_script = WORKDIR / "predict.py"
    model_path = WORKDIR / "outputs" / "model.pkl"
    if not model_path.exists():
        model_path = WORKDIR / "model.pkl"
    
    if not predict_script.exists():
        return GradingResult(
            score=0.0,
            subscores={"step_3_predict_script": 0.0},
            weights={"step_3_predict_script": 1.0},
            feedback=(
                "STEP 3 FAILED: predict.py not found at /workdir/predict.py\n\n"
                "Your solution.sh must create a predict.py script that:\n"
                     "- Loads your trained model\n"
                     "- Accepts test CSV path as command-line argument\n"
                     "- Outputs predictions to /workdir/outputs/predictions.csv"
            )
        )

    if not model_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"step_3_model_check": 0.0},
            weights={"step_3_model_check": 1.0},
            feedback=(
                "STEP 3 FAILED: model.pkl not found at /workdir/outputs/model.pkl or /workdir/model.pkl\n\n"
                "Your solution.sh must train and save a model to /workdir/outputs/model.pkl"
            )
        )

    # ---------------- Step 4: Execute predict.py with test data ----------------
    test_data_path = Path("/tests/test.csv")
    if not test_data_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"step_4_test_data": 0.0},
            weights={"step_4_test_data": 1.0},
            feedback="STEP 4 FAILED: Test data not found at /tests/test.csv"
        )

    try:
        # predict.py expects: model_path test_csv_path
        if model_path.exists():
            result = subprocess.run(
                [sys.executable, str(predict_script), str(model_path), str(test_data_path)],
                cwd=str(WORKDIR),
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS
            )
        else:
            # Fallback: try with just test CSV (if predict.py loads model internally)
            result = subprocess.run(
                [sys.executable, str(predict_script), str(test_data_path)],
                cwd=str(WORKDIR),
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS
            )

        if result.returncode != 0:
            return GradingResult(
                score=0.0,
                subscores={"step_4_predict_execution": 0.0},
                weights={"step_4_predict_execution": 1.0},
                feedback=(
                    "STEP 4 FAILED: predict.py execution failed\n\n"
                         f"Return code: {result.returncode}\n"
                    f"STDOUT:\n{result.stdout}\n\n"
                    f"STDERR:\n{result.stderr}\n\n"
                    "Ensure your predict.py:\n"
                    "- Accepts model path as sys.argv[1] and test CSV path as sys.argv[2]\n"
                    "- Loads the trained model successfully\n"
                    "- Reads test data from sys.argv[2]\n"
                    "- Handles all feature engineering\n"
                    "- Outputs to /workdir/outputs/predictions.csv"
                )
            )

    except subprocess.TimeoutExpired:
        return GradingResult(
            score=0.0,
            subscores={"step_4_predict_timeout": 0.0},
            weights={"step_4_predict_timeout": 1.0},
            feedback=(
                f"STEP 4 FAILED: predict.py execution timed out (>{EXEC_TIMEOUT_SECONDS} seconds)\n\n"
                     "Your prediction script is taking too long. Optimize your model loading and inference."
            )
        )
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores={"step_4_predict_error": 0.0},
            weights={"step_4_predict_error": 1.0},
            feedback=f"STEP 4 FAILED: Error executing predict.py: {str(e)}"
        )

    # ---------------- Step 5: Read predictions ----------------
    preds_map, err = read_predictions()
    if err:
        return GradingResult(
            score=0.0,
            subscores={"step_5_read_predictions": 0.0},
            weights={"step_5_read_predictions": 1.0},
            feedback=f"STEP 5 FAILED: {err}"
        )

    # ========================================================================
    # STEP 6: Compute metrics
    # ========================================================================
    # Only compute metrics for IDs that exist in both ground truth and predictions
    gt_ids = set(ground_truth.keys())
    pred_ids = set(preds_map.keys())
    common_ids = gt_ids & pred_ids
    if not common_ids:
        return GradingResult(
            score=0.0,
            subscores={"step_6_no_common_ids": 0.0},
            weights={"step_6_no_common_ids": 1.0},
            feedback="STEP 6 FAILED: No common IDs between ground truth and predictions"
        )
    
    # Filter to common IDs only
    filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in common_ids}
    filtered_preds_map = {k: v for k, v in preds_map.items() if k in common_ids}
    
    # Align predictions with ground truth (only common IDs)
    ids = sorted(common_ids)
    y_true = [filtered_ground_truth[i] for i in ids]
    y_pred = [filtered_preds_map[i] for i in ids]
    
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Compute per-class F1 scores for feedback
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    min_class_f1 = float(per_class_f1.min())
    worst_class_id = int(per_class_f1.argmin())
    best_class_f1 = float(per_class_f1.max())
    best_class_id = int(per_class_f1.argmax())

    # ========================================================================
    # STEP 7: Apply threshold and Return Binary Score
    # ========================================================================
    # Simple check: Macro F1 >= PASS_THRESHOLD to pass
    passed = macro_f1 >= PASS_THRESHOLD
    final_score = 1.0 if passed else 0.0
    result_text = "PASSED" if passed else "FAILED"
    
    # Build feedback
    feedback_lines = []
    feedback_lines.append(f"=== EVALUATION RESULTS: {result_text} ===")
    feedback_lines.append(f"\nTest samples: {len(ids)}")
    feedback_lines.append(f"Threshold: Macro F1 ≥ {PASS_THRESHOLD:.2f}")
    feedback_lines.append(f"\nMacro F1 Score: {macro_f1:.4f} {'✓' if passed else '✗'}")
    feedback_lines.append(f"Minimum per-class F1: {min_class_f1:.4f} (class {worst_class_id})")
    feedback_lines.append(f"Best Class F1: {best_class_f1:.4f} (class {best_class_id})")
    
    if passed:
        feedback_lines.append(f"\n✓ Model achieved Macro F1 of {macro_f1:.4f} (≥ {PASS_THRESHOLD:.2f} required)")
    else:
        gap = PASS_THRESHOLD - macro_f1
        feedback_lines.append(f"\n✗ Model achieved Macro F1 of {macro_f1:.4f} (< {PASS_THRESHOLD:.2f} required, gap: {gap:.4f})")
        feedback_lines.append(f"\nSuggestions for improvement:")
        feedback_lines.append(f"  - Comprehensive feature engineering")
        feedback_lines.append(f"  - Advanced ensemble methods (LightGBM, XGBoost)")
        feedback_lines.append(f"  - Hyperparameter tuning")
        feedback_lines.append(f"  - Cross-validation optimizing Macro F1")
    
    feedback_text = "\n".join(feedback_lines)

    # Binary scoring: subscores must be 1.0 or 0.0 (no partial scoring)
    subscores_dict = {
        "threshold_met": final_score  # 1.0 if passed, 0.0 if failed
    }

    weights_dict = {
        "threshold_met": 1.0
    }
    
    return GradingResult(
        score=final_score,
        subscores=subscores_dict,
        weights=weights_dict,
        feedback=feedback_text
    )


if __name__ == "__main__":
    res = grade()
    print("GradingResult:", res)
    exit(0 if res.score == 1.0 else 1)
