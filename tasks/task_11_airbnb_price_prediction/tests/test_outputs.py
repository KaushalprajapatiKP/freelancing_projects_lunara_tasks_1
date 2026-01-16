from pathlib import Path
import csv
import subprocess
import sys
import pandas as pd
import numpy as np


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    mse = sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)
    rmse = np.sqrt(mse)
    return rmse


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
        assert 'price' in fieldnames or 'Price' in fieldnames or 'PRICE' in fieldnames, (
            f"Invalid header. Expected 'price' column, got {fieldnames}"
        )
        
        id_col = 'id' if 'id' in fieldnames else ('ID' if 'ID' in fieldnames else 'Id')
        price_col = 'price' if 'price' in fieldnames else ('Price' if 'Price' in fieldnames else 'PRICE')
        
        predictions = []
        for row in reader:
            try:
                pred_value = float(row[price_col])
                # Price should be non-negative
                assert pred_value >= 0, (
                    f"Prediction {row[id_col]}: price {pred_value} must be >= 0"
                )
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
        price_col = 'price' if 'price' in fieldnames else ('Price' if 'Price' in fieldnames else 'PRICE')
        predictions = [(row[id_col], float(row[price_col])) for row in reader]
    
    pred_dict = {str(pid): pval for pid, pval in predictions}
    
    # Get id column from ground truth (handle different case variations)
    gt_id_col = None
    for col in ['id', 'ID', 'Id']:
        if col in gt_df.columns:
            gt_id_col = col
            break
    
    assert gt_id_col is not None, "Could not find id column in ground truth"
    
    # Convert IDs to string for consistent matching
    gt_ids = set(gt_df[gt_id_col].astype(str))
    pred_ids = set(pred_dict.keys())
    
    assert len(gt_df) == len(predictions), (
        f"Prediction count mismatch. Expected {len(gt_df)}, got {len(predictions)}"
    )
    
    missing_ids = gt_ids - pred_ids
    assert len(missing_ids) == 0, (
        f"Missing predictions for {len(missing_ids)} ids. First few: {list(missing_ids)[:5]}"
    )
    
    extra_ids = pred_ids - gt_ids
    assert len(extra_ids) == 0, (
        f"Extra predictions for {len(extra_ids)} unknown ids. First few: {list(extra_ids)[:5]}"
    )


def test_rmse_threshold():
    """Check that RMSE meets the threshold requirement"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_ground_truth.csv"
    predictions_path = Path("/app/predictions.csv")
    
    print(f"\n{'='*60}")
    print(f"Running RMSE Threshold Test")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(ground_truth_path)
    pred_df = pd.read_csv(predictions_path)
    
    # Get id and price columns from ground truth
    gt_id_col = None
    for col in ['id', 'ID', 'Id']:
        if col in gt_df.columns:
            gt_id_col = col
            break
    
    assert gt_id_col is not None, "Could not find id column in ground truth"
    
    # Get price column from ground truth
    gt_price_col = None
    for col in ['price', 'Price', 'PRICE', 'target', 'TARGET', 'y']:
        if col in gt_df.columns:
            gt_price_col = col
            break
    
    if gt_price_col is None:
        # Assume last column is price
        gt_price_col = gt_df.columns[-1]
    
    # Get id and price columns from predictions
    pred_id_col = None
    for col in ['id', 'ID', 'Id']:
        if col in pred_df.columns:
            pred_id_col = col
            break
    
    assert pred_id_col is not None, "Could not find id column in predictions"
    
    pred_price_col = None
    for col in ['price', 'Price', 'PRICE', 'target', 'TARGET', 'y']:
        if col in pred_df.columns:
            pred_price_col = col
            break
    
    assert pred_price_col is not None, "Could not find price column in predictions"
    
    # Select only the columns we need for merging and comparison
    gt_subset = gt_df[[gt_id_col, gt_price_col]].copy()
    pred_subset = pred_df[[pred_id_col, pred_price_col]].copy()
    
    # Rename columns to avoid conflicts
    gt_subset = gt_subset.rename(columns={gt_price_col: 'price_true'})
    pred_subset = pred_subset.rename(columns={pred_price_col: 'price_pred'})
    
    # Convert IDs to string for consistent matching
    gt_subset[gt_id_col] = gt_subset[gt_id_col].astype(str)
    pred_subset[pred_id_col] = pred_subset[pred_id_col].astype(str)
    
    # Merge on IDs
    merged = gt_subset.merge(pred_subset, left_on=gt_id_col, right_on=pred_id_col, how='inner')
    
    assert len(merged) > 0, f"No matching predictions found - IDs don't match between ground truth and predictions. GT IDs sample: {gt_subset[gt_id_col].head().tolist()}, Pred IDs sample: {pred_subset[pred_id_col].head().tolist()}"
    
    y_true = merged['price_true'].tolist()
    y_pred = merged['price_pred'].tolist()
    
    assert len(y_true) > 0, "No matching predictions found"
    
    print(f"Loaded {len(y_true)} ground truth and prediction pairs")
    
    # Calculate RMSE
    rmse = calculate_rmse(y_true, y_pred)
    
    # Threshold: RMSE ≤ 111.0
    threshold_rmse = 111.0
    
    # Print metrics to logs
    print(f"\n=== RMSE Metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE Threshold: {threshold_rmse:.4f}")
    print(f"RMSE Status: {'PASS' if rmse <= threshold_rmse else 'FAIL'}")
    print(f"Number of samples: {len(y_true)}")
    print(f"Mean actual price: {sum(y_true)/len(y_true):.4f}")
    print(f"Mean predicted price: {sum(y_pred)/len(y_pred):.4f}")
    print(f"Min actual: {min(y_true):.4f}, Max actual: {max(y_true):.4f}")
    print(f"Min predicted: {min(y_pred):.4f}, Max predicted: {max(y_pred):.4f}")
    print("=" * 40)
    
    assert rmse <= threshold_rmse, (
        f"RMSE {rmse:.4f} is above threshold {threshold_rmse:.4f}"
    )

