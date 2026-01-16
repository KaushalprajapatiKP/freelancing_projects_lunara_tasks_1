from pathlib import Path
import csv
import subprocess
import sys
import pandas as pd


def test_predict_script_exists():
    """Check if predict.py exists in /app/"""
    predict_script = Path("/app/predict.py")
    assert predict_script.exists(), "predict.py not found at /app/predict.py"


def test_predict_script_executes():
    """Execute predict.py with test data"""
    testdir = Path("/tests")
    test_data_path = testdir / "test.csv"
    predict_script = Path("/app/predict.py")
    
    assert test_data_path.exists(), f"Test data file not found at {test_data_path}"
    
    result = subprocess.run(
        ["/usr/bin/python3", str(predict_script), str(test_data_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="/app"
    )
    
    assert result.returncode == 0, f"predict.py execution failed: {result.stderr[:500]}"


def test_predictions_file_exists():
    """Check if predictions.csv exists"""
    predictions_path = Path("/app/predictions.csv")
    assert predictions_path.exists(), "predictions.csv not found at /app/predictions.csv"


def test_predictions_format():
    """Validate predictions.csv format"""
    predictions_path = Path("/app/predictions.csv")
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['id', 'label'], (
            f"Invalid header. Expected ['id', 'label'], got {reader.fieldnames}"
        )
        
        valid_labels = {'Rise', 'Fall', 'Neutral'}
        predictions = []
        for row in reader:
            label = row['label']
            assert label in valid_labels, (
                f"Invalid label '{label}' for id {row['id']}. Must be one of: {valid_labels}"
            )
            predictions.append((row['id'], label))


def test_predictions_completeness():
    """Check that all test samples have predictions"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    assert ground_truth_path.exists(), f"Ground truth file not found at {ground_truth_path}"
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['id'], row['label']) for row in reader]
    
    pred_dict = {pid: plabel for pid, plabel in predictions}
    
    assert len(gt_df) == len(predictions), (
        f"Prediction count mismatch. Expected {len(gt_df)}, got {len(predictions)}"
    )
    
    missing_ids = set(gt_df['id'].astype(str)) - set(pred_dict.keys())
    assert len(missing_ids) == 0, (
        f"Missing predictions for {len(missing_ids)} IDs. First few: {list(missing_ids)[:5]}"
    )
    
    extra_ids = set(pred_dict.keys()) - set(gt_df['id'].astype(str))
    assert len(extra_ids) == 0, (
        f"Extra predictions for {len(extra_ids)} unknown IDs. First few: {list(extra_ids)[:5]}"
    )


def test_accuracy_threshold():
    """Check that Accuracy meets the threshold requirement"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    print(f"\n{'='*60}")
    print(f"Running Accuracy Threshold Test")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['id'], row['label']) for row in reader]
    
    pred_dict = {pid: plabel for pid, plabel in predictions}
    
    y_true = []
    y_pred = []
    
    for _, row in gt_df.iterrows():
        gt_id = str(row['id'])
        if gt_id in pred_dict:
            y_true.append(row['label'])
            y_pred.append(pred_dict[gt_id])
    
    assert len(y_true) > 0, "No matching predictions found"
    
    print(f"Loaded {len(y_true)} ground truth and prediction pairs")
    
    # Calculate Accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
    
    # Calculate score: 100% if accuracy ≥ 0.70, 0% if accuracy ≤ 0.40, linear in between
    acc_best, acc_worst = 0.70, 0.40
    if accuracy >= acc_best:
        acc_score = 1.0
    elif accuracy <= acc_worst:
        acc_score = 0.0
    else:
        acc_score = (accuracy - acc_worst) / (acc_best - acc_worst)
    
    # Primary threshold: Accuracy ≥ 0.453 (45.3%)
    threshold_acc = 0.453
    
    # Print metrics to logs
    print(f"\n=== Accuracy Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy Score: {acc_score*100:.2f}%")
    print(f"Accuracy Threshold: {threshold_acc:.4f}")
    print(f"Accuracy Status: {'PASS' if accuracy >= threshold_acc else 'FAIL'}")
    print(f"Required accuracy threshold: >= {threshold_acc:.4f}")
    print(f"Number of samples: {len(y_true)}")
    print(f"Correct predictions: {correct}")
    
    assert accuracy >= threshold_acc, (
        f"Accuracy {accuracy:.4f} is below threshold {threshold_acc:.4f} "
        f"(score: {acc_score*100:.2f}%)"
    )

