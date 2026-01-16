from pathlib import Path
import csv
import subprocess
import sys
import pandas as pd


def calculate_r2_score(y_true, y_pred):
    """Calculate R² (Coefficient of Determination)"""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # Calculate mean of actual values
    mean_actual = sum(y_true) / len(y_true)
    
    # Sum of squares of residuals (SS_res)
    ss_res = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    
    # Total sum of squares (SS_tot)
    ss_tot = sum((a - mean_actual) ** 2 for a in y_true)
    
    # R² = 1 - (SS_res / SS_tot)
    if ss_tot == 0:
        return 0.0  # All values are the same, cannot calculate R² meaningfully
    
    r2 = 1 - (ss_res / ss_tot)
    return r2


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
        fieldnames = list(reader.fieldnames)
        assert 'id' in fieldnames or 'ID' in fieldnames or 'Id' in fieldnames, (
            f"Invalid header. Expected 'id' column, got {fieldnames}"
        )
        assert 'target' in fieldnames or 'Target' in fieldnames or 'TARGET' in fieldnames, (
            f"Invalid header. Expected 'target' column, got {fieldnames}"
        )
        
        id_col = 'id' if 'id' in fieldnames else ('ID' if 'ID' in fieldnames else 'Id')
        target_col = 'target' if 'target' in fieldnames else ('Target' if 'Target' in fieldnames else 'TARGET')
        
        predictions = []
        for row in reader:
            try:
                pred_value = float(row[target_col])
                # Target should be a numeric value (allow negative values for regression)
                predictions.append((row[id_col], pred_value))
            except (ValueError, KeyError) as e:
                raise AssertionError(f"Invalid prediction row: {row}") from e


def test_predictions_completeness():
    """Check that all test ids have predictions"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    assert ground_truth_path.exists(), f"Ground truth file not found at {ground_truth_path}"
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        id_col = 'id' if 'id' in fieldnames else ('ID' if 'ID' in fieldnames else 'Id')
        target_col = 'target' if 'target' in fieldnames else ('Target' if 'Target' in fieldnames else 'TARGET')
        predictions = [(row[id_col], float(row[target_col])) for row in reader]
    
    pred_dict = {str(pid): pval for pid, pval in predictions}
    
    # Get id column from ground truth (handle different case variations)
    gt_id_col = None
    for col in ['id', 'ID', 'Id']:
        if col in gt_df.columns:
            gt_id_col = col
            break
    
    assert gt_id_col is not None, "Could not find id column in ground truth"
    
    assert len(gt_df) == len(predictions), (
        f"Prediction count mismatch. Expected {len(gt_df)}, got {len(predictions)}"
    )
    
    missing_ids = set(gt_df[gt_id_col].astype(str)) - set(pred_dict.keys())
    assert len(missing_ids) == 0, (
        f"Missing predictions for {len(missing_ids)} ids. First few: {list(missing_ids)[:5]}"
    )
    
    extra_ids = set(pred_dict.keys()) - set(gt_df[gt_id_col].astype(str))
    assert len(extra_ids) == 0, (
        f"Extra predictions for {len(extra_ids)} unknown ids. First few: {list(extra_ids)[:5]}"
    )


def test_r2_threshold():
    """Check that R² meets the threshold requirement"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    print(f"\n{'='*60}")
    print(f"Running R² Threshold Test")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(ground_truth_path)
    pred_df = pd.read_csv(predictions_path)
    
    # Get id and target columns from ground truth
    gt_id_col = None
    for col in ['id', 'ID', 'Id']:
        if col in gt_df.columns:
            gt_id_col = col
            break
    
    assert gt_id_col is not None, "Could not find id column in ground truth"
    
    # Get target column from ground truth
    gt_target_col = None
    for col in ['target', 'Target', 'TARGET', 'y']:
        if col in gt_df.columns:
            gt_target_col = col
            break
    
    if gt_target_col is None:
        # Assume last column is target
        gt_target_col = gt_df.columns[-1]
    
    # Get id and target columns from predictions
    pred_id_col = None
    for col in ['id', 'ID', 'Id']:
        if col in pred_df.columns:
            pred_id_col = col
            break
    
    assert pred_id_col is not None, "Could not find id column in predictions"
    
    pred_target_col = None
    for col in ['target', 'Target', 'TARGET']:
        if col in pred_df.columns:
            pred_target_col = col
            break
    
    assert pred_target_col is not None, "Could not find target column in predictions"
    
    # Create dictionaries for easier lookup
    gt_dict = {str(row[gt_id_col]): row[gt_target_col] for _, row in gt_df.iterrows()}
    pred_dict = {str(row[pred_id_col]): row[pred_target_col] for _, row in pred_df.iterrows()}
    
    # Match IDs and extract corresponding target values
    y_true = []
    y_pred = []
    
    for gt_id in gt_dict.keys():
        if gt_id in pred_dict:
            y_true.append(gt_dict[gt_id])
            y_pred.append(pred_dict[gt_id])
    
    assert len(y_true) > 0, "No matching predictions found - IDs don't match between ground truth and predictions"
    
    print(f"Loaded {len(y_true)} ground truth and prediction pairs")
    
    # Calculate R²
    r2 = calculate_r2_score(y_true, y_pred)
    
    # Threshold: R² ≥ 0.38 (reasonable threshold for regression tasks)
    threshold_r2 = 0.38
    
    # Print metrics to logs
    print(f"\n=== R² Metrics ===")
    print(f"R² Score: {r2:.4f}")
    print(f"R² Threshold: {threshold_r2:.4f}")
    print(f"R² Status: {'PASS' if r2 >= threshold_r2 else 'FAIL'}")
    print(f"Number of samples: {len(y_true)}")
    print(f"Mean actual target: {sum(y_true)/len(y_true):.4f}")
    print(f"Mean predicted target: {sum(y_pred)/len(y_pred):.4f}")
    print(f"Min actual: {min(y_true):.4f}, Max actual: {max(y_true):.4f}")
    print(f"Min predicted: {min(y_pred):.4f}, Max predicted: {max(y_pred):.4f}")
    print("=" * 40)
    
    assert r2 >= threshold_r2, (
        f"R² {r2:.4f} is below threshold {threshold_r2:.4f}"
    )

