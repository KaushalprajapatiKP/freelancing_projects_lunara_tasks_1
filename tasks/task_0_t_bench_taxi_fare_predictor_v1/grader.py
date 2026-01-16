#!/usr/bin/env python3
"""
Grader for Taxi Fare Prediction (Model Code Submission)

Sequential validation with binary scoring (0.0 or 1.0):
1. Check if predict.py exists in /workdir/
2. Execute predict.py with test data path
3. Validate predictions.csv output format
4. Calculate MAE, RMSE, R² scores
5. Apply 96% threshold and return binary score (1 if ≥96%, 0 otherwise)
"""

from pathlib import Path
import csv
import subprocess
import sys
from apex_arena._types import GradingResult


def grade(_ctx=None) -> GradingResult:
    """
    Grade the taxi fare prediction task.

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
    test_data_path = Path("/tests/test_trips.csv")

    if not test_data_path.exists():
        return GradingResult(
            score=0.0,
            subscores={"test_data_check": 0.0},
            weights={"test_data_check": 1.0},
            feedback="STEP 2 FAILED: Test data not found at /tests/test_trips.csv"
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

    ground_truth = {}
    try:
        with ground_truth_path.open('r') as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != ['trip_id', 'fare_amount']:
                return GradingResult(
                    score=0.0,
                    subscores={"ground_truth_format": 0.0},
                    weights={"ground_truth_format": 1.0},
                    feedback=f"STEP 3 FAILED: Ground truth has incorrect header: {reader.fieldnames}"
                )

            for row in reader:
                trip_id = row['trip_id']
                fare = float(row['fare_amount'])
                ground_truth[trip_id] = fare

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

            if reader.fieldnames != ['trip_id', 'predicted_fare']:
                return GradingResult(
                    score=0.0,
                    subscores={"predictions_header": 0.0},
                    weights={"predictions_header": 1.0},
                    feedback=f"STEP 4 FAILED: predictions.csv has incorrect header\n"
                             f"Expected: ['trip_id', 'predicted_fare']\n"
                             f"Got: {reader.fieldnames}"
                )

            for row_num, row in enumerate(reader, start=2):
                trip_id = row['trip_id']

                # Check if trip_id exists in ground truth
                if trip_id not in ground_truth:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_trip_id": 0.0},
                        weights={"predictions_trip_id": 1.0},
                        feedback=f"STEP 4 FAILED: Unknown trip_id '{trip_id}' at row {row_num}"
                    )

                # Parse predicted fare
                try:
                    predicted_fare = float(row['predicted_fare'])
                except ValueError:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_parse": 0.0},
                        weights={"predictions_parse": 1.0},
                        feedback=f"STEP 4 FAILED: Invalid predicted_fare '{row['predicted_fare']}' at row {row_num}\n"
                                 f"Must be a numeric value"
                    )

                # Check if fare is positive
                if predicted_fare <= 0:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_positive": 0.0},
                        weights={"predictions_positive": 1.0},
                        feedback=f"STEP 4 FAILED: predicted_fare must be > 0, got {predicted_fare} at row {row_num}"
                    )

                # Check for duplicates
                if trip_id in predictions:
                    return GradingResult(
                        score=0.0,
                        subscores={"predictions_duplicates": 0.0},
                        weights={"predictions_duplicates": 1.0},
                        feedback=f"STEP 4 FAILED: Duplicate trip_id '{trip_id}' at row {row_num}"
                    )

                predictions[trip_id] = predicted_fare

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
            feedback += f"Missing trip_ids: {list(missing)[:10]}\n"
        if extra:
            feedback += f"Extra trip_ids: {list(extra)[:10]}\n"

        return GradingResult(
            score=0.0,
            subscores={"predictions_completeness": 0.0},
            weights={"predictions_completeness": 1.0},
            feedback=feedback
        )

    # ========================================================================
    # STEP 6: Calculate Metrics
    # ========================================================================
    errors = []
    squared_errors = []
    actual_values = []
    predicted_values = []

    for trip_id in ground_truth:
        actual = ground_truth[trip_id]
        predicted = predictions[trip_id]

        error = abs(actual - predicted)
        squared_error = (actual - predicted) ** 2

        errors.append(error)
        squared_errors.append(squared_error)
        actual_values.append(actual)
        predicted_values.append(predicted)

    # Calculate metrics
    mae = sum(errors) / len(errors)
    print(f"MAE: {mae}")
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    print(f"RMSE: {rmse}")

    # Calculate R²
    mean_actual = sum(actual_values) / len(actual_values)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual_values)
    ss_res = sum(squared_errors)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    print(f"R²: {r2}")
    # ========================================================================
    # STEP 7: Score Metrics
    # ========================================================================
    # MAE scoring: 100% if ≤ $3.50, 0% if ≥ $7.00, linear in between
    mae_best, mae_worst = 3.5, 7.0
    if mae <= mae_best:
        mae_score = 1.0
    elif mae >= mae_worst:
        mae_score = 0.0
    else:
        mae_score = (mae_worst - mae) / (mae_worst - mae_best)

    # Overall score (for now, just MAE)
    overall_score = mae_score
    print(f"Overall score: {overall_score*100:.2f}%")

    # ========================================================================
    # STEP 8: Apply Threshold and Return Binary Score
    # ========================================================================
    THRESHOLD = 0.96

    if overall_score >= THRESHOLD:
        final_score = 1.0
        result_text = "PASSED"
    else:
        final_score = 0.0
        result_text = "FAILED"

    # Build feedback
    feedback_lines = []
    feedback_lines.append(f"=== EVALUATION RESULTS: {result_text} ===")
    feedback_lines.append(f"\nTest samples: {n_test}")
    feedback_lines.append(f"Required threshold: {THRESHOLD*100:.0f}%")
    feedback_lines.append(f"Overall score: {overall_score*100:.2f}%")

    feedback_lines.append(f"\n--- Prediction Metrics ---")
    feedback_lines.append(f"MAE (Mean Absolute Error): ${mae:.2f}")
    feedback_lines.append(f"  Score: {mae_score*100:.1f}% (100% if ≤$3.50, 0% if ≥$7.00)")
    feedback_lines.append(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
    feedback_lines.append(f"R² Score: {r2:.4f}")

    feedback_lines.append(f"\n--- Error Distribution ---")
    feedback_lines.append(f"Min error: ${min(errors):.2f}")
    feedback_lines.append(f"Max error: ${max(errors):.2f}")
    feedback_lines.append(f"Median error: ${sorted(errors)[len(errors)//2]:.2f}")

    if final_score == 1.0:
        feedback_lines.append(f"\n✓ Model achieved {overall_score*100:.2f}% score (≥ {THRESHOLD*100:.0f}% required)")
        feedback_lines.append(f"  MAE of ${mae:.2f} indicates good prediction accuracy")
    else:
        feedback_lines.append(f"\n✗ Model achieved {overall_score*100:.2f}% score (< {THRESHOLD*100:.0f}% required)")
        feedback_lines.append(f"  MAE of ${mae:.2f} is too high - target ≤ $3.50 for full score")
        feedback_lines.append(f"\nSuggestions for improvement:")
        feedback_lines.append(f"  - Better feature engineering (temporal, geospatial)")
        feedback_lines.append(f"  - Use ensemble methods (GradientBoosting, RandomForest)")
        feedback_lines.append(f"  - Proper feature scaling and encoding")
        feedback_lines.append(f"  - Include interaction features (e.g., distance * surge)")

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
    