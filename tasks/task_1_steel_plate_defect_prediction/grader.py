#!/usr/bin/env python3
"""
Grader for Steel Plate Defect Prediction (Multi-label Classification)

Sequential validation with binary scoring (0.0 or 1.0):
1. Check if predict.py exists in /workdir/
2. Execute predict.py with test data path
3. Validate predictions.csv output format
4. Calculate ROC-AUC scores for each of 7 defect types and average them
5. Apply 90% threshold and return binary score (1 if ≥90%, 0 otherwise)
"""

from pathlib import Path
import csv
import subprocess
import sys
from apex_arena._types import GradingResult
from sklearn.metrics import roc_auc_score
import numpy as np


def grade(_ctx=None) -> GradingResult:
    """
    Grade the steel plate defect prediction task.
    
    Returns:
        GradingResult with score 1.0 if ROC-AUC >= 0.90, else 0.0
    """
    
    # ========================================================================
    # STEP 1: Check if predict.py exists
    # ========================================================================
    predict_script = Path("/workdir/predict.py")
    
    if not predict_script.exists():
        return GradingResult(
            score=0.0,
            subscores={"predict_script_check": 0.0},
            weights={"predict_script_check": 1.0},
            feedback="STEP 1 FAILED: predict.py not found at /workdir/predict.py\n\n"
                     "You must create a Python script that:\n"
                     "- Loads your trained model\n"
                     "- Accepts test CSV path as command-line argument\n"
                     "- Outputs predictions to /workdir/predictions.csv"
        )
    
    # ========================================================================
    # STEP 2: Execute predict.py with test data
    # ========================================================================
    test_data_path = Path("/tests/test.csv")
    
    if not test_data_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"test_data_check": 0.0},
            weights={"test_data_check": 1.0},
            feedback="STEP 2 FAILED: Test data not found at /tests/test.csv"
        )
    
    try:
        # Execute predict.py with test data path
        result = subprocess.run(
            [sys.executable, str(predict_script), str(test_data_path)],
            cwd="/workdir",
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            return GradingResult(
                score=0.0,
                subscores={"predict_execution": 0.0},
                weights={"predict_execution": 1.0},
                feedback=f"STEP 2 FAILED: predict.py execution failed\n\n"
                         f"Return code: {result.returncode}\n"
                         f"STDOUT:\n{result.stdout}\n\n"
                         f"STDERR:\n{result.stderr}\n\n"
                         f"Ensure your predict.py:\n"
                         f"- Loads the trained model successfully\n"
                         f"- Reads test data from sys.argv[1]\n"
                         f"- Handles all feature engineering\n"
                         f"- Outputs to /workdir/predictions.csv"
            )
    
    except subprocess.TimeoutExpired:
        return GradingResult(
            score=0.0,
            subscores={"predict_timeout": 0.0},
            weights={"predict_timeout": 1.0},
            feedback="STEP 2 FAILED: predict.py execution timed out (>120 seconds)\n\n"
                     "Your prediction script is taking too long. Optimize your model loading and inference."
        )
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores={"predict_error": 0.0},
            weights={"predict_error": 1.0},
            feedback=f"STEP 2 FAILED: Error executing predict.py: {str(e)}"
        )
    
    # ========================================================================
    # STEP 3: Load Ground Truth
    # ========================================================================
    ground_truth_path = Path("/tests/test_ground_truth.csv")
    
    if not ground_truth_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"ground_truth_check": 0.0},
            weights={"ground_truth_check": 1.0},
            feedback="STEP 3 FAILED: Ground truth file not found at /tests/test_ground_truth.csv"
        )
    
    target_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    ground_truth = {}
    
    try:
        with ground_truth_path.open('r') as f:
            reader = csv.DictReader(f)
            
            expected_columns = ['id'] + target_columns
            if set(reader.fieldnames) != set(expected_columns):
                return GradingResult(
                    score=0.0,
                    subscores={"ground_truth_format": 0.0},
                    weights={"ground_truth_format": 1.0},
                    feedback=f"STEP 3 FAILED: Ground truth has incorrect columns\n"
                             f"Expected: {expected_columns}\n"
                             f"Got: {reader.fieldnames}"
                )
            
            for row in reader:
                plate_id = row['id']
                ground_truth[plate_id] = {}
                for target in target_columns:
                    ground_truth[plate_id][target] = int(row[target])
    
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores={"ground_truth_load": 0.0},
            weights={"ground_truth_load": 1.0},
            feedback=f"STEP 3 FAILED: Error loading ground truth: {str(e)}"
        )
    
    n_test = len(ground_truth)
    
    # ========================================================================
    # STEP 4: Load and Validate Predictions
    # ========================================================================
    predictions_path = Path("/workdir/predictions.csv")
    
    if not predictions_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"predictions_exist": 0.0},
            weights={"predictions_exist": 1.0},
            feedback="STEP 4 FAILED: predictions.csv not found at /workdir/predictions.csv\n\n"
                     "Your predict.py must output predictions to /workdir/predictions.csv"
        )
    
    predictions = {}
    try:
        with predictions_path.open('r') as f:
            reader = csv.DictReader(f)
            
            expected_columns = ['id'] + target_columns
            if set(reader.fieldnames) != set(expected_columns):
                return GradingResult(
                    score=0.0,
                    subscores={"predictions_header": 0.0},
                    weights={"predictions_header": 1.0},
                    feedback=f"STEP 4 FAILED: predictions.csv has incorrect header\n"
                             f"Expected: {expected_columns}\n"
                             f"Got: {reader.fieldnames}"
                )
            
            for row_num, row in enumerate(reader, start=2):
                plate_id = row['id']
                
                # Check if plate_id exists in ground truth
                if plate_id not in ground_truth:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_plate_id": 0.0},
                        weights={"predictions_plate_id": 1.0},
                        feedback=f"STEP 4 FAILED: Unknown plate_id '{plate_id}' at row {row_num}"
                    )
                
                # Check for duplicates
                if plate_id in predictions:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_duplicates": 0.0},
                        weights={"predictions_duplicates": 1.0},
                        feedback=f"STEP 4 FAILED: Duplicate plate_id '{plate_id}' at row {row_num}"
                    )
                
                # Parse predicted probabilities
                predictions[plate_id] = {}
                for target in target_columns:
                    try:
                        prob = float(row[target])
                        # Validate probability range
                        if prob < 0.0 or prob > 1.0:
                            return GradingResult(
                                score=0.0,
                                subscores={"predictions_range": 0.0},
                                weights={"predictions_range": 1.0},
                                feedback=f"STEP 4 FAILED: Invalid probability {prob} for {target} at row {row_num}\n"
                                         f"Probabilities must be between 0.0 and 1.0"
                            )
                        predictions[plate_id][target] = prob
                    except ValueError:
                        return GradingResult(
                            score=0.0,
                            subscores={"predictions_parse": 0.0},
                            weights={"predictions_parse": 1.0},
                            feedback=f"STEP 4 FAILED: Invalid probability '{row[target]}' for {target} at row {row_num}\n"
                                     f"Must be a numeric value"
                        )
    
    except Exception as e:
        return GradingResult(
            score=0.0,
            subscores={"predictions_load": 0.0},
            weights={"predictions_load": 1.0},
            feedback=f"STEP 4 FAILED: Error loading predictions: {str(e)}"
        )
    
    # ========================================================================
    # STEP 5: Check Completeness
    # ========================================================================
    if len(predictions) != n_test:
        missing = set(ground_truth.keys()) - set(predictions.keys())
        extra = set(predictions.keys()) - set(ground_truth.keys())
        
        feedback = f"STEP 5 FAILED: Expected {n_test} predictions, got {len(predictions)}\n"
        if missing:
            feedback += f"Missing plate_ids: {list(missing)[:10]}\n"
        if extra:
            feedback += f"Extra plate_ids: {list(extra)[:10]}\n"
        
        return GradingResult(
            score=0.0,
            subscores={"predictions_completeness": 0.0},
            weights={"predictions_completeness": 1.0},
            feedback=feedback
        )
    
    # ========================================================================
    # STEP 6: Calculate ROC-AUC for each target and average
    # ========================================================================
    # Ensure consistent order
    plate_ids = sorted(ground_truth.keys())
    
    roc_auc_scores = []
    target_roc_aucs = {}
    
    for target in target_columns:
        y_true = np.array([ground_truth[pid][target] for pid in plate_ids])
        y_pred = np.array([predictions[pid][target] for pid in plate_ids])
        
        # Calculate ROC-AUC (handle case where all labels are same class)
        try:
            if len(np.unique(y_true)) == 1:
                # All same class - cannot compute ROC-AUC meaningfully
                # Use 0.5 as default (random classifier performance)
                roc_auc = 0.5
            else:
                roc_auc = roc_auc_score(y_true, y_pred)
            roc_auc_scores.append(roc_auc)
            target_roc_aucs[target] = roc_auc
        except Exception as e:
            return GradingResult(
                score=0.0,
                subscores={"roc_auc_calculation": 0.0},
                weights={"roc_auc_calculation": 1.0},
                feedback=f"STEP 6 FAILED: Error calculating ROC-AUC for {target}: {str(e)}"
            )
    
    # Average ROC-AUC across all targets
    mean_roc_auc = np.mean(roc_auc_scores)
    print(f"Mean ROC-AUC: {mean_roc_auc:.6f}")
    
    # ========================================================================
    # STEP 7: Apply Thresholds and Return Binary Score
    # ========================================================================
    # Primary threshold: mean ROC-AUC >= 0.8835 to pass
    ROC_AUC_THRESHOLD = 0.8835
    # Secondary threshold: minimum per-target ROC-AUC >= 0.70
    MIN_TARGET_THRESHOLD = 0.70
    # Fairness constraint: max variance in per-target ROC-AUC <= 0.10
    MAX_AUC_VARIANCE = 0.10
    
    # Check primary threshold
    primary_pass = mean_roc_auc >= ROC_AUC_THRESHOLD
    
    # Check secondary threshold (minimum per-target score)
    min_target_score = min(target_roc_aucs.values())
    secondary_pass = min_target_score >= MIN_TARGET_THRESHOLD
    
    # Check fairness constraint (AUC variance)
    auc_values = list(target_roc_aucs.values())
    auc_variance = np.var(auc_values)
    fairness_pass = auc_variance <= MAX_AUC_VARIANCE
    
    # All three checks must pass
    if primary_pass and secondary_pass and fairness_pass:
        final_score = 1.0
        result_text = "PASSED"
    else:
        final_score = 0.0
        result_text = "FAILED"
    
    # Build feedback
    feedback_lines = []
    feedback_lines.append(f"=== EVALUATION RESULTS: {result_text} ===")
    feedback_lines.append(f"\nTest samples: {n_test}")
    feedback_lines.append(f"Primary threshold: ROC-AUC ≥ {ROC_AUC_THRESHOLD}")
    feedback_lines.append(f"Secondary threshold: Min per-target ROC-AUC ≥ {MIN_TARGET_THRESHOLD}")
    feedback_lines.append(f"Fairness constraint: Max AUC variance ≤ {MAX_AUC_VARIANCE}")
    feedback_lines.append(f"\nMean ROC-AUC: {mean_roc_auc:.6f}")
    feedback_lines.append(f"Minimum per-target ROC-AUC: {min_target_score:.6f}")
    feedback_lines.append(f"AUC Variance: {auc_variance:.6f}")
    
    feedback_lines.append(f"\n--- Per-Target ROC-AUC Scores ---")
    for target, score in target_roc_aucs.items():
        status = "✓" if score >= MIN_TARGET_THRESHOLD else "✗"
        feedback_lines.append(f"{status} {target}: {score:.6f}")
    
    if final_score == 1.0:
        feedback_lines.append(f"\n✓ Model achieved ROC-AUC of {mean_roc_auc:.6f} (≥ {ROC_AUC_THRESHOLD} required)")
        feedback_lines.append(f"✓ Minimum per-target ROC-AUC: {min_target_score:.6f} (≥ {MIN_TARGET_THRESHOLD} required)")
        feedback_lines.append(f"✓ AUC variance: {auc_variance:.6f} (≤ {MAX_AUC_VARIANCE} required)")
        feedback_lines.append(f"\n  Excellent multi-label classification with balanced performance across all defect types")
    else:
        if not primary_pass:
            feedback_lines.append(f"\n✗ Model achieved ROC-AUC of {mean_roc_auc:.6f} (< {ROC_AUC_THRESHOLD} required)")
        if not secondary_pass:
            feedback_lines.append(f"\n✗ Minimum per-target ROC-AUC: {min_target_score:.6f} (< {MIN_TARGET_THRESHOLD} required)")
        if not fairness_pass:
            feedback_lines.append(f"\n✗ AUC variance: {auc_variance:.6f} (> {MAX_AUC_VARIANCE} allowed)")
            feedback_lines.append(f"   Performance is too unbalanced across targets. Focus on improving weak targets.")
        feedback_lines.append(f"\nSuggestions for improvement:")
        feedback_lines.append(f"  - Advanced feature engineering (ratios, interactions, polynomial features)")
        feedback_lines.append(f"  - Strong ensemble methods (XGBoost, LightGBM with proper tuning)")
        feedback_lines.append(f"  - Multi-label approaches (Classifier Chains, stacking)")
        feedback_lines.append(f"  - Aggressive handling of class imbalance (oversampling, class weights)")
        feedback_lines.append(f"  - Per-target model optimization for weak classes")
        feedback_lines.append(f"  - Calibration techniques for better probability estimates")
    
    return GradingResult(
        score=final_score,
        subscores={"overall_check": final_score},
        weights={"overall_check": 1.0},
        feedback="\n".join(feedback_lines)
    )


if __name__ == "__main__":
    import json
    result = grade(None)
    print(result.model_dump_json())

