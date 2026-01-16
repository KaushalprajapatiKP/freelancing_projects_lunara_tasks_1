from pathlib import Path
import csv
import subprocess
import sys
import pandas as pd


def calculate_auc_roc(y_true, y_pred):
    sorted_data = sorted(zip(y_pred, y_true), reverse=True)
    
    total_positive = sum(y_true)
    total_negative = len(y_true) - total_positive
    
    if total_positive == 0 or total_negative == 0:
        return 0.5
    
    auc = 0.0
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    
    for pred, true_label in sorted_data:
        if true_label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / total_positive if total_positive > 0 else 0.0
        fpr = fp / total_negative if total_negative > 0 else 0.0
        
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr
    
    return auc


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
        assert reader.fieldnames == ['id', 'target'], (
            f"Invalid header. Expected ['id', 'target'], got {reader.fieldnames}"
        )
        
        predictions = []
        for row in reader:
            try:
                pred_value = float(row['target'])
                assert 0.0 <= pred_value <= 1.0, (
                    f"Prediction {row['id']}: value {pred_value} not in [0.0, 1.0]"
                )
                predictions.append((row['id'], pred_value))
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
        predictions = [(row['id'], float(row['target'])) for row in reader]
    
    pred_dict = {pid: pval for pid, pval in predictions}
    
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


def test_auc_roc_threshold():
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['id'], float(row['target'])) for row in reader]
    
    pred_dict = {pid: pval for pid, pval in predictions}
    
    y_true = []
    y_pred = []
    
    for _, row in gt_df.iterrows():
        gt_id = str(row['id'])
        if gt_id in pred_dict:
            y_val = row['Y']
            y_true.append(1 if y_val else 0)
            y_pred.append(pred_dict[gt_id])
    
    assert len(y_true) > 0, "No matching predictions found"
    
    for pred in y_pred:
        assert 0.0 <= pred <= 1.0, f"Invalid probability value: {pred}"
    
    auc = calculate_auc_roc(y_true, y_pred)
    threshold = 0.787
    auc_rounded = round(auc, 4)
    threshold_rounded = round(threshold, 4)
    
    assert auc_rounded >= threshold_rounded, (
        f"AUC-ROC {auc_rounded:.4f} is below threshold {threshold_rounded:.4f}"
    )

