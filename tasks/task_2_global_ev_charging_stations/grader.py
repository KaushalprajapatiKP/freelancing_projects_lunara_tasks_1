"""
grader.py — EV Charging Stations Operational Status model  

Purpose
-------
This grader validates submissions for the EV Charging Stations Operational Status task.
It implements a strict, sequential evaluation protocol: the first failed check causes an
immediate FAIL (score 0.0). Only when all checks pass does the grader compute the macro-F1
on the protected test set and return PASS (score 1.0) if the metric meets or exceeds the
configured threshold.

Security & Isolation
--------------------
- The grader assumes a secure environment where the agent cannot read `/tests/`.
- Ground-truth labels live in `/tests/test_labels.csv` and are *never* exposed to the agent.
- The agent may only operate on `/data/` (training) and `/workdir` (its workspace).
- The grader expects the agent's solution to have already run and produced `/workdir/outputs/predictions.csv`.

High-level evaluation flow
--------------------------
1. Load ground truth from `/tests/test_labels.csv`. Must contain header: id,status.
2. Verify that the participant provided `/workdir/solution.sh`.
3. Read predictions file if available (if not, use empty predictions).
4. Validate each predicted value for syntactic correctness (if predictions exist).
5. Compute Macro-F1 between ground truth and predictions (only on common IDs).
6. Compare Macro-F1 to PASS_THRESHOLD and return a binary GradingResult.

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
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# The GradingResult dataclass (platform-provided) — expected return object
from apex_arena._types import GradingResult

# Metric computation
from sklearn.metrics import f1_score, classification_report

# -----------------------------
# Configuration constants
# -----------------------------
# Location of the canonical (secure) ground truth file.
GROUND_TRUTH_PATH = Path("/tests/test_labels.csv")

# Workspace where agent runs. The grader executes the agent's entrypoint inside /workdir.
WORKDIR = Path("/workdir")

# Agent's required entrypoint script. The grader will run this script (bash).
SOLUTION_SCRIPT = WORKDIR / "solution.sh"

# Expected location of predictions after agent runs.
# The grader expects the CSV to be exactly at /workdir/outputs/predictions.csv
PREDICTIONS_PATH = WORKDIR / "outputs" / "predictions.csv"

# Allowed label set for the classification task; must match grader & task contract exactly.
ALLOWED_LABELS = {"Operational", "Not Operational"}

# Execution timeout for the participant's solution. If exceeded, the grader fails.
# Must match max_agent_timeout_sec in task.yaml metadata
EXEC_TIMEOUT_SECONDS = 7200  # 2 hours

# Macro F1 threshold to pass the task. Must be consistent with task.yaml and grader logic.
PASS_THRESHOLD = 0.54

# -----------------------------
# Utility: file reading helpers
# -----------------------------
def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Load the ground truth mapping from id -> status from a secure CSV.

    Expected CSV header: id,status

    Returns:
      - (dict, None) on success where dict maps id -> 'Operational'|'Not Operational'
      - (None, error_message) on failure
    """
    if not path.exists():
        return None, f"Ground truth file not found at {path}"

    try:
        gt: Dict[str, str] = {}
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Strict header check so the grader can detect misformatted ground-truths early.
            if not reader.fieldnames or "id" not in reader.fieldnames or "status" not in reader.fieldnames:
                return None, "Ground truth CSV must include header columns: 'id' and 'status'."
            for row in reader:
                idx = row["id"]
                status = row["status"]
                # Ground truth must itself be canonical — if it isn't, that's a grader-side data error.
                if status not in ALLOWED_LABELS:
                    return None, f"Ground truth contains non-canonical label '{status}' for id '{idx}'."
                gt[idx] = status
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


def read_predictions(path: Path = PREDICTIONS_PATH) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Read participant predictions from CSV and return a mapping id -> prediction.

    Expected CSV header EXACTLY: id,prediction (in that order).
    The function returns (preds_map, None) on success or (None, error_message) on failure.
    """
    if not path.exists():
        return None, f"Predictions file not found at {path}"

    try:
        preds: Dict[str, str] = {}
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            expected = ["id", "prediction"]
            # The header must match exactly to avoid ambiguity about column ordering or naming.
            if reader.fieldnames != expected:
                return None, f"Invalid CSV header. Expected: {expected}. Got: {reader.fieldnames}"
            row_no = 0
            for row in reader:
                row_no += 1
                idx = row["id"]
                pred = row["prediction"]
                if not idx:
                    return None, f"Empty id at CSV row {row_no}"
                # Save predicted label as-is (we will validate canonicalness later)
                preds[idx] = pred
        if not preds:
            return None, "Predictions CSV is empty."
        return preds, None
    except Exception as exc:
        return None, f"Failed to read predictions CSV: {exc}"


# -----------------------------
# Validation utilities
# -----------------------------
def validate_allowed_labels(preds: Dict[str, str], allowed: set = ALLOWED_LABELS) -> Tuple[bool, Optional[str]]:
    """
    Ensure every predicted value is a canonical label defined in ALLOWED_LABELS.

    Returns (True, None) if valid, otherwise (False, error_message).
    """
    for idx, v in preds.items():
        if v not in allowed:
            return False, f"Invalid prediction value '{v}' for id '{idx}'. Allowed labels: {sorted(list(allowed))}"
    return True, None


# -----------------------------
# Metric utilities
# -----------------------------
def compute_macro_f1(ground_truth: Dict[str, str], predictions: Dict[str, str]) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute macro-averaged F1 score between ground truth and predictions.

    Returns (macro_f1, None) on success, (None, error_message) on failure.
    """
    try:
        # Align order: sort ids to keep deterministic reporting
        ids = sorted(ground_truth.keys())
        y_true = [1 if ground_truth[i] == "Operational" else 0 for i in ids]
        y_pred = [1 if predictions[i] == "Operational" else 0 for i in ids]

        macro = f1_score(y_true, y_pred, average="macro")
        return float(macro), None
    except Exception as exc:
        return None, f"Error computing macro F1: {exc}"


# -----------------------------
# Primary grading function
# -----------------------------
def grade(_ctx=None) -> GradingResult:
    """
    Primary grader entry point.

    Sequential checks:
      1) Load ground truth
      2) Check solution.sh exists
      3) Run predict.py to generate predictions (if predict.py and test data exist)
      4) Read predictions file
      5) Validate canonical label values (if predictions exist)
      6) Compute macro-F1 on common IDs and compare with PASS_THRESHOLD

    On first failure the function returns a GradingResult with score=0.0 and a helpful feedback message.
    On success, returns score=1.0 and feedback including the metric details.
    """
    # ---------------- Step 1: Load ground truth ----------------
    ground_truth, err = load_ground_truth()
    if err:
        return GradingResult(
            score=0.0,
            subscores={"step_1_load_ground_truth": 0.0},
            weights={"step_1_load_ground_truth": 1.0},
            feedback=f"STEP 1 FAILED: {err}"
        )

    # ---------------- Step 2: Check solution.sh presence ----------------
    if not SOLUTION_SCRIPT.exists():
        return GradingResult(
            score=0.0,
            subscores={"step_2_solution_script_exists": 0.0},
            weights={"step_2_solution_script_exists": 1.0},
            feedback=f"STEP 2 FAILED: solution script not found at {SOLUTION_SCRIPT}"
        )

    # ---------------- Step 3: Check if predict.py exists and run it to generate predictions ----------------
    predict_script = WORKDIR / "predict.py"
    test_csv_path = Path("/tests/test.csv")
    
    if predict_script.exists() and test_csv_path.exists():
        # Run predict.py to generate predictions
        try:
            result = subprocess.run(
                [sys.executable, str(predict_script), str(test_csv_path)],
                cwd=str(WORKDIR),
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS
            )
            if result.returncode != 0:
                # If predict.py fails, continue anyway - we'll try to read existing predictions
                pass
        except Exception:
            # If execution fails, continue anyway - we'll try to read existing predictions
            pass

    # ---------------- Step 4: Read predictions file ----------------
    preds_map, err = read_predictions()
    if err:
        # If predictions file doesn't exist or can't be read, use empty dict
        # This will result in score 0.0 but won't fail as a step error
        preds_map = {}

    # ---------------- Step 5: Validate canonical label values (if predictions exist) ----------------
    if preds_map:
        ok, err = validate_allowed_labels(preds_map)
        if not ok:
            return GradingResult(
                score=0.0,
                subscores={"step_5_invalid_labels": 0.0},
                weights={"step_5_invalid_labels": 1.0},
                feedback=f"STEP 5 FAILED: {err}"
            )

    # ---------------- Step 6: Compute macro-F1 and decide pass/fail ----------------
    # Only compute metrics for IDs that exist in both ground truth and predictions
    gt_ids = set(ground_truth.keys())
    pred_ids = set(preds_map.keys())
    common_ids = gt_ids & pred_ids
    
    # Always compute F1 score - use common IDs if available, otherwise 0.0
    if not common_ids:
        # No common IDs: F1 = 0.0
        macro_f1 = 0.0
        has_class_report = False
        class_report = None
        num_common = 0
    else:
        # Filter to common IDs only and compute metrics
        filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in common_ids}
        filtered_preds_map = {k: v for k, v in preds_map.items() if k in common_ids}
        num_common = len(common_ids)
        
        macro_f1, err = compute_macro_f1(filtered_ground_truth, filtered_preds_map)
        if err:
            # If metric computation fails, use 0.0
            macro_f1 = 0.0
            has_class_report = False
            class_report = None
        else:
            # Prepare human-friendly classification report for feedback
            sorted_ids = sorted(common_ids)
            y_true = [1 if filtered_ground_truth[i] == "Operational" else 0 for i in sorted_ids]
            y_pred = [1 if filtered_preds_map[i] == "Operational" else 0 for i in sorted_ids]
            class_report = classification_report(y_true, y_pred, target_names=["Not Operational", "Operational"], zero_division=0)
            has_class_report = True

    passed = macro_f1 >= PASS_THRESHOLD
    final_score = 1.0 if passed else 0.0
    status = "PASS" if passed else "FAIL"

    # ---------------- Final feedback message ----------------
    feedback_lines: List[str] = []
    feedback_lines.append(f"STEP 7 RESULT: {status}")
    feedback_lines.append(f"Macro F1 Score: {macro_f1:.4f}")
    feedback_lines.append(f"Pass threshold: {PASS_THRESHOLD:.2f}")
    feedback_lines.append(f"Common IDs: {num_common} / {len(gt_ids)} (ground truth) / {len(pred_ids)} (predictions)")
    
    if has_class_report and class_report:
        feedback_lines.append("")
        feedback_lines.append("Classification report (per-class):")
        feedback_lines.append(class_report)
    elif not common_ids:
        feedback_lines.append("")
        feedback_lines.append("Note: No common IDs between ground truth and predictions, so F1 = 0.0")
    
    feedback_lines.append("")
    feedback_lines.append("Notes:")
    feedback_lines.append("- Metrics computed only on common IDs between ground truth and predictions")
    feedback_lines.append(f"- Predictions file: {PREDICTIONS_PATH}")
    feedback_text = "\n".join(feedback_lines)

    return GradingResult(
        score=final_score,
        subscores={"macro_f1_check": final_score},
        weights={"macro_f1_check": 1.0},
        feedback=feedback_text
    )


# If invoked directly, run a local check (useful for manual grader debugging).
if __name__ == "__main__":
    # When running this file locally (not in the grading harness), we print the result for convenience.
    res = grade()
    # The grading harness expects the object, but for CLI we print a summary and exit with 0/1.
    print("GradingResult:", res)
    # Best-effort exit code: 0=pass, 1=fail
    exit(0 if res.score == 1.0 else 1)
