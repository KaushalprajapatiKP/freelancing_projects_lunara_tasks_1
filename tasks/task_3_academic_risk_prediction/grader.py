#!/usr/bin/env python3
"""
Grader for Academic Risk Prediction (Model Code Submission)

Sequential validation with binary scoring (0.0 or 1.0):
1. Check if predict.py exists in /workdir/
2. Execute predict.py with test data path
3. Validate predictions.csv output format
4. Calculate Accuracy and other classification metrics
5. Apply 96% threshold and return binary score (1 if ≥96%, 0 otherwise)
"""

from pathlib import Path
import csv
import subprocess
import sys
from apex_arena._types import GradingResult


def grade(_ctx=None) -> GradingResult:
    """
    Grade the academic risk prediction task.

    Returns:
        GradingResult with score 1.0 if score >= 96%, else 0.0
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
        # Execute predict.py with model path and test data path
        model_path = Path("/workdir/model.pkl")
        if not model_path.exists():
            return GradingResult(
                score=0.0,
                subscores={"model_check": 0.0},
                weights={"model_check": 1.0},
                feedback="STEP 2 FAILED: model.pkl not found at /workdir/model.pkl\n\n"
                         "Your solution.sh must train and save a model to /workdir/model.pkl"
            )
        
        result = subprocess.run(
            [sys.executable, str(predict_script), str(model_path), str(test_data_path)],
            cwd="/workdir",
            capture_output=True,
            text=True,
            timeout=300
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
                         f"- Accepts model path as sys.argv[1] and test CSV path as sys.argv[2]\n"
                         f"- Loads the trained model from the provided path\n"
                         f"- Validates test data format and required columns\n"
                         f"- Applies the same feature engineering used during training\n"
                         f"- Handles missing values appropriately\n"
                         f"- Outputs to /workdir/predictions.csv"
            )

    except subprocess.TimeoutExpired:
        return GradingResult(
            score=0.0,
            subscores={"predict_timeout": 0.0},
            weights={"predict_timeout": 1.0},
            feedback="STEP 2 FAILED: predict.py execution timed out (>300 seconds)\n\n"
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

    ground_truth = {}
    valid_targets = {"Graduate", "Enrolled", "Dropout"}
    
    try:
        with ground_truth_path.open('r') as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != ['id', 'Target']:
                return GradingResult(
                    score=0.0,
                    subscores={"ground_truth_format": 0.0},
                    weights={"ground_truth_format": 1.0},
                    feedback=f"STEP 3 FAILED: Ground truth has incorrect header: {reader.fieldnames}\n"
                             f"Expected: ['id', 'Target']"
                )

            for row in reader:
                student_id = row['id']
                target = row['Target']
                
                if target not in valid_targets:
                    return GradingResult(
                        score=0.0,
                        subscores={"ground_truth_invalid": 0.0},
                        weights={"ground_truth_invalid": 1.0},
                        feedback=f"STEP 3 FAILED: Invalid target value '{target}' in ground truth\n"
                                 f"Valid targets: {valid_targets}"
                    )
                
                ground_truth[student_id] = target

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

            if reader.fieldnames != ['id', 'Target']:
                return GradingResult(
                    score=0.0,
                    subscores={"predictions_header": 0.0},
                    weights={"predictions_header": 1.0},
                    feedback=f"STEP 4 FAILED: predictions.csv has incorrect header\n"
                             f"Expected: ['id', 'Target']\n"
                             f"Got: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):
                student_id = row['id']
                predicted_target = row['Target']

                # Check if student_id exists in ground truth
                if student_id not in ground_truth:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_student_id": 0.0},
                        weights={"predictions_student_id": 1.0},
                        feedback=f"STEP 4 FAILED: Unknown student_id '{student_id}' at row {row_num}"
                    )

                # Validate target value
                if predicted_target not in valid_targets:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_invalid": 0.0},
                        weights={"predictions_invalid": 1.0},
                        feedback=f"STEP 4 FAILED: Invalid Target value '{predicted_target}' at row {row_num}\n"
                                 f"Valid targets: {valid_targets}"
                    )

                # Check for duplicates
                if student_id in predictions:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_duplicates": 0.0},
                        weights={"predictions_duplicates": 1.0},
                        feedback=f"STEP 4 FAILED: Duplicate student_id '{student_id}' at row {row_num}"
                    )

                predictions[student_id] = predicted_target

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
            feedback += f"Missing student_ids: {list(missing)[:10]}\n"
        if extra:
            feedback += f"Extra student_ids: {list(extra)[:10]}\n"

        return GradingResult(
            score=0.0,
            subscores={"predictions_completeness": 0.0},
            weights={"predictions_completeness": 1.0},
            feedback=feedback
        )

    # ========================================================================
    # STEP 6: Calculate Metrics
    # ========================================================================
    correct = 0
    total = 0
    class_correct = {"Graduate": 0, "Enrolled": 0, "Dropout": 0}
    class_total = {"Graduate": 0, "Enrolled": 0, "Dropout": 0}
    confusion_matrix = {target: {pred: 0 for pred in valid_targets} for target in valid_targets}

    for student_id in ground_truth:
        actual = ground_truth[student_id]
        predicted = predictions[student_id]

        class_total[actual] += 1
        confusion_matrix[actual][predicted] += 1

        if actual == predicted:
            correct += 1
            class_correct[actual] += 1
        total += 1

    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.6f}")

    # Calculate per-class metrics
    class_accuracy = {}
    for target in valid_targets:
        if class_total[target] > 0:
            class_accuracy[target] = class_correct[target] / class_total[target]
        else:
            class_accuracy[target] = 0.0

    # Calculate precision, recall, F1 for each class
    class_metrics = {}
    for target in valid_targets:
        tp = confusion_matrix[target][target]
        fp = sum(confusion_matrix[pred][target] for pred in valid_targets if pred != target)
        fn = sum(confusion_matrix[target][pred] for pred in valid_targets if pred != target)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[target] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Macro-averaged metrics
    macro_precision = sum(m['precision'] for m in class_metrics.values()) / len(class_metrics)
    macro_recall = sum(m['recall'] for m in class_metrics.values()) / len(class_metrics)
    macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(class_metrics)

    print(f"Macro Precision: {macro_precision:.6f}")
    print(f"Macro Recall: {macro_recall:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")

    # ========================================================================
    # STEP 7: Binary Pass/Fail Evaluation
    # ========================================================================
    # Binary threshold: 83.25% accuracy
    ACCURACY_THRESHOLD = 0.8325

    # Check accuracy threshold (binary: pass or fail only)
    accuracy_passed = accuracy >= ACCURACY_THRESHOLD
    
    # Calculate minimum per-class F1 for reporting (not a threshold)
    min_f1_class = None
    min_f1_value = 1.0
    for target in valid_targets:
        if class_metrics[target]['f1'] < min_f1_value:
            min_f1_value = class_metrics[target]['f1']
            min_f1_class = target
    
    # Binary result: Pass or Fail only (no partial scoring)
    if accuracy_passed:
        final_score = 1.0
        result_text = "PASSED"
    else:
        final_score = 0.0
        result_text = "FAILED"

    # Build feedback
    feedback_lines = []
    feedback_lines.append(f"=== EVALUATION RESULTS: {result_text} ===")
    feedback_lines.append(f"\nTest samples: {n_test}")
    feedback_lines.append(f"Required threshold:")
    feedback_lines.append(f"  - Accuracy: ≥ {ACCURACY_THRESHOLD*100:.1f}%")

    feedback_lines.append(f"\n--- Prediction Metrics ---")
    feedback_lines.append(f"Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    accuracy_status = "✓ PASSED" if accuracy_passed else "✗ FAILED"
    feedback_lines.append(f"  Status: {accuracy_status} (Required: ≥ {ACCURACY_THRESHOLD*100:.1f}%)")
    feedback_lines.append(f"Macro Precision: {macro_precision:.6f}")
    feedback_lines.append(f"Macro Recall: {macro_recall:.6f}")
    feedback_lines.append(f"Macro F1-Score: {macro_f1:.6f} ({macro_f1*100:.2f}%)")
    feedback_lines.append(f"Minimum Per-Class F1: {min_f1_value:.6f} ({min_f1_value*100:.2f}%) [{min_f1_class}]")

    feedback_lines.append(f"\n--- Per-Class Metrics ---")
    for target in sorted(valid_targets):
        metrics = class_metrics[target]
        acc = class_accuracy[target]
        feedback_lines.append(f"{target}:")
        feedback_lines.append(f"  Accuracy: {acc:.4f} ({class_correct[target]}/{class_total[target]})")
        feedback_lines.append(f"  Precision: {metrics['precision']:.4f}")
        feedback_lines.append(f"  Recall: {metrics['recall']:.4f}")
        feedback_lines.append(f"  F1-Score: {metrics['f1']:.4f}")

    feedback_lines.append(f"\n--- Confusion Matrix ---")
    feedback_lines.append("Actual \\ Predicted | " + " | ".join(f"{p:>10}" for p in sorted(valid_targets)))
    feedback_lines.append("-" * (15 + 13 * len(valid_targets)))
    for actual in sorted(valid_targets):
        row = f"{actual:>14} | " + " | ".join(f"{confusion_matrix[actual][pred]:>10}" for pred in sorted(valid_targets))
        feedback_lines.append(row)

    if final_score == 1.0:
        feedback_lines.append(f"\n✓ Model PASSED the accuracy threshold:")
        feedback_lines.append(f"  - Accuracy: {accuracy*100:.2f}% (≥ {ACCURACY_THRESHOLD*100:.1f}% required)")
        feedback_lines.append(f"  Excellent prediction performance!")
    else:
        feedback_lines.append(f"\n✗ Model FAILED - accuracy threshold not met:")
        feedback_lines.append(f"  - Accuracy: {accuracy*100:.2f}% (< {ACCURACY_THRESHOLD*100:.1f}% required)")
        feedback_lines.append(f"\nSuggestions for improvement:")
        feedback_lines.append(f"  - Use stacking with meta-learners (LogisticRegression on top of base models)")
        feedback_lines.append(f"  - Apply probability calibration (CalibratedClassifierCV with isotonic/sigmoid)")
        feedback_lines.append(f"  - Advanced feature engineering (approval rate trends, failure patterns, interaction features)")
        feedback_lines.append(f"  - Ensemble multiple diverse models (LightGBM, XGBoost with different hyperparameters)")

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

