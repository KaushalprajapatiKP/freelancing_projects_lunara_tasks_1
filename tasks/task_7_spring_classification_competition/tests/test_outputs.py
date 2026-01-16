from pathlib import Path
import csv
import subprocess
import sys
import pandas as pd


def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1-Score from true labels and predicted probabilities.
    For imbalanced datasets, finds optimal threshold that maximizes F1-Score.
    """
    # Find optimal threshold for F1-Score (try multiple thresholds)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
        
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_binary[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_binary[i] == 1)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_binary[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    return best_f1


def test_predict_script_exists():
    predict_script = Path("/app/predict.py")
    assert predict_script.exists(), "predict.py not found at /app/predict.py"


def test_predict_script_executes():
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
    predictions_path = Path("/app/predictions.csv")
    assert predictions_path.exists(), "predictions.csv not found at /app/predictions.csv"


def test_predictions_format():
    predictions_path = Path("/app/predictions.csv")
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        assert 'ID' in fieldnames or 'id' in fieldnames, (
            f"Invalid header. Expected 'ID' or 'id' column, got {fieldnames}"
        )
        assert 'TARGET' in fieldnames or 'target' in fieldnames, (
            f"Invalid header. Expected 'TARGET' or 'target' column, got {fieldnames}"
        )
        
        id_col = 'ID' if 'ID' in fieldnames else 'id'
        target_col = 'TARGET' if 'TARGET' in fieldnames else 'target'
        
        predictions = []
        for row in reader:
            try:
                pred_value = float(row[target_col])
                assert 0.0 <= pred_value <= 1.0, (
                    f"Prediction {row[id_col]}: value {pred_value} not in [0.0, 1.0]"
                )
                predictions.append((row[id_col], pred_value))
            except (ValueError, KeyError) as e:
                raise AssertionError(f"Invalid prediction row: {row}") from e


def test_predictions_completeness():
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    assert ground_truth_path.exists(), f"Ground truth file not found at {ground_truth_path}"
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        id_col = 'ID' if 'ID' in fieldnames else 'id'
        target_col = 'TARGET' if 'TARGET' in fieldnames else 'target'
        predictions = [(row[id_col], float(row[target_col])) for row in reader]
    
    pred_dict = {str(pid): pval for pid, pval in predictions}
    
    # Get ID column from ground truth
    gt_id_col = 'ID' if 'ID' in gt_df.columns else 'id'
    gt_target_col = 'stroke' if 'stroke' in gt_df.columns else ('TARGET' if 'TARGET' in gt_df.columns else 'target')
    
    assert len(gt_df) == len(predictions), (
        f"Prediction count mismatch. Expected {len(gt_df)}, got {len(predictions)}"
    )
    
    missing_ids = set(gt_df[gt_id_col].astype(str)) - set(pred_dict.keys())
    assert len(missing_ids) == 0, (
        f"Missing predictions for {len(missing_ids)} IDs. First few: {list(missing_ids)[:5]}"
    )
    
    extra_ids = set(pred_dict.keys()) - set(gt_df[gt_id_col].astype(str))
    assert len(extra_ids) == 0, (
        f"Extra predictions for {len(extra_ids)} unknown IDs. First few: {list(extra_ids)[:5]}"
    )


def test_f1_score_threshold():
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        id_col = 'ID' if 'ID' in fieldnames else 'id'
        target_col = 'TARGET' if 'TARGET' in fieldnames else 'target'
        predictions = [(row[id_col], float(row[target_col])) for row in reader]
    
    pred_dict = {str(pid): pval for pid, pval in predictions}
    
    # Get ID and target columns from ground truth
    gt_id_col = 'ID' if 'ID' in gt_df.columns else 'id'
    gt_target_col = 'stroke' if 'stroke' in gt_df.columns else ('TARGET' if 'TARGET' in gt_df.columns else 'target')
    
    y_true = []
    y_pred = []
    
    for _, row in gt_df.iterrows():
        gt_id = str(row[gt_id_col])
        if gt_id in pred_dict:
            y_val = row[gt_target_col]
            y_true.append(1 if y_val == 1 or y_val is True else 0)
            y_pred.append(pred_dict[gt_id])
    
    assert len(y_true) > 0, "No matching predictions found"
    
    for pred in y_pred:
        assert 0.0 <= pred <= 1.0, f"Invalid probability value: {pred}"
    
    # Log actual values for debugging
    print(f"\n=== F1-Score Evaluation Details ===")
    print(f"Total samples: {len(y_true)}")
    print(f"Positive class in ground truth: {sum(y_true)} ({sum(y_true)/len(y_true)*100:.2f}%)")
    print(f"Prediction statistics:")
    print(f"  Min probability: {min(y_pred):.4f}")
    print(f"  Max probability: {max(y_pred):.4f}")
    print(f"  Mean probability: {sum(y_pred)/len(y_pred):.4f}")
    print(f"  Median probability: {sorted(y_pred)[len(y_pred)//2]:.4f}")
    print(f"  Predictions >= 0.5: {sum(1 for p in y_pred if p >= 0.5)}")
    print(f"  Predictions >= 0.3: {sum(1 for p in y_pred if p >= 0.3)}")
    print(f"  Predictions >= 0.2: {sum(1 for p in y_pred if p >= 0.2)}")
    print(f"  Predictions >= 0.1: {sum(1 for p in y_pred if p >= 0.1)}")
    
    # Show some example predictions
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(y_true))):
        print(f"  ID: {list(gt_df[gt_id_col])[i]}, True: {y_true[i]}, Pred: {y_pred[i]:.4f}")
    
    f1 = calculate_f1_score(y_true, y_pred)
    threshold = 0.305
    f1_rounded = round(f1, 4)
    threshold_rounded = round(threshold, 4)
    
    print(f"\nF1-Score: {f1_rounded:.4f}")
    print(f"Threshold required: {threshold_rounded:.4f}")
    print(f"Status: {'PASS' if f1_rounded >= threshold_rounded else 'FAIL'}")
    print("=" * 40)
    
    assert f1_rounded >= threshold_rounded, (
        f"F1-Score {f1_rounded:.4f} is below threshold {threshold_rounded:.4f}"
    )

