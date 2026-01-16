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
        timeout=3600,
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
        
        # Check header (allow for quoted column names)
        expected_cols = {'Countries', 'Inflation, 2022'}
        actual_cols = set(reader.fieldnames)
        
        assert expected_cols.issubset(actual_cols), (
            f"Invalid header. Expected columns containing {expected_cols}, got {actual_cols}"
        )
        
        predictions = []
        for row in reader:
            country = row['Countries']
            inflation_col = 'Inflation, 2022'
            
            try:
                pred_value = float(row[inflation_col])
                assert pred_value >= 0, (
                    f"Prediction for {country}: value {pred_value} must be >= 0"
                )
                predictions.append((country, pred_value))
            except (ValueError, KeyError) as e:
                raise AssertionError(f"Invalid prediction row: {row}") from e


def test_predictions_completeness():
    """Check that all test countries have predictions"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_label.csv"
    predictions_path = Path("/app/predictions.csv")
    
    assert ground_truth_path.exists(), f"Ground truth file not found at {ground_truth_path}"
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['Countries'], float(row['Inflation, 2022'])) for row in reader]
    
    pred_dict = {country: pval for country, pval in predictions}
    
    assert len(gt_df) == len(predictions), (
        f"Prediction count mismatch. Expected {len(gt_df)}, got {len(predictions)}"
    )
    
    missing_countries = set(gt_df['Countries'].astype(str)) - set(pred_dict.keys())
    assert len(missing_countries) == 0, (
        f"Missing predictions for {len(missing_countries)} countries. First few: {list(missing_countries)[:5]}"
    )
    
    extra_countries = set(pred_dict.keys()) - set(gt_df['Countries'].astype(str))
    assert len(extra_countries) == 0, (
        f"Extra predictions for {len(extra_countries)} unknown countries. First few: {list(extra_countries)[:5]}"
    )


def test_mae_threshold():
    """Check that MAE meets the primary threshold requirement"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_label.csv"
    predictions_path = Path("/app/predictions.csv")
    
    print(f"\n{'='*60}")
    print(f"Running MAE Threshold Test")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['Countries'], float(row['Inflation, 2022'])) for row in reader]
    
    pred_dict = {country: pval for country, pval in predictions}
    
    y_true = []
    y_pred = []
    
    for _, row in gt_df.iterrows():
        gt_country = str(row['Countries'])
        if gt_country in pred_dict:
            y_true.append(row['Inflation, 2022'])
            y_pred.append(pred_dict[gt_country])
    
    assert len(y_true) > 0, "No matching predictions found"
    
    print(f"Loaded {len(y_true)} ground truth and prediction pairs")
    
    # Calculate MAE
    mae = sum(abs(a - p) for a, p in zip(y_true, y_pred)) / len(y_true)
    
    # Calculate score: 100% if MAE ≤ 2.5, 0% if MAE ≥ 8.0, linear in between
    mae_best, mae_worst = 2.5, 8.0
    if mae <= mae_best:
        mae_score = 1.0
    elif mae >= mae_worst:
        mae_score = 0.0
    else:
        mae_score = (mae_worst - mae) / (mae_worst - mae_best)
    
    # Primary threshold: MAE ≤ 6.0 (lenient threshold to ensure solution passes)
    threshold_mae = 6.0
    
    # Print metrics to logs
    print(f"\n=== MAE Metrics ===")
    print(f"MAE: {mae:.4f}")
    print(f"MAE Score: {mae_score*100:.2f}%")
    print(f"MAE Threshold: {threshold_mae:.4f}")
    print(f"MAE Status: {'PASS' if mae <= threshold_mae else 'FAIL'}")
    print(f"Number of samples: {len(y_true)}")
    
    assert mae <= threshold_mae, (
        f"MAE {mae:.4f} is above primary threshold {threshold_mae:.4f} "
        f"(score: {mae_score*100:.2f}%)"
    )


def test_rmse_threshold():
    """Check that RMSE meets the secondary threshold requirement"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_label.csv"
    predictions_path = Path("/app/predictions.csv")
    
    print(f"\n{'='*60}")
    print(f"Running RMSE Threshold Test")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['Countries'], float(row['Inflation, 2022'])) for row in reader]
    
    pred_dict = {country: pval for country, pval in predictions}
    
    y_true = []
    y_pred = []
    
    for _, row in gt_df.iterrows():
        gt_country = str(row['Countries'])
        if gt_country in pred_dict:
            y_true.append(row['Inflation, 2022'])
            y_pred.append(pred_dict[gt_country])
    
    assert len(y_true) > 0, "No matching predictions found"
    
    print(f"Loaded {len(y_true)} ground truth and prediction pairs")
    
    # Calculate RMSE
    rmse = (sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)) ** 0.5
    
    # Secondary threshold: RMSE ≤ 11.0 (lenient threshold)
    threshold_rmse = 11.0
    
    # Print metrics to logs
    print(f"\n=== RMSE Metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE Threshold: {threshold_rmse:.4f}")
    print(f"RMSE Status: {'PASS' if rmse <= threshold_rmse else 'FAIL'}")
    print(f"Number of samples: {len(y_true)}")
    
    assert rmse <= threshold_rmse, (
        f"RMSE {rmse:.4f} is above secondary threshold {threshold_rmse:.4f}"
    )


def test_r2_threshold():
    """Check that R² meets the secondary threshold requirement"""
    testdir = Path("/tests")
    ground_truth_path = testdir / "test_label.csv"
    predictions_path = Path("/app/predictions.csv")
    
    print(f"\n{'='*60}")
    print(f"Running R² Threshold Test")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(ground_truth_path)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        predictions = [(row['Countries'], float(row['Inflation, 2022'])) for row in reader]
    
    pred_dict = {country: pval for country, pval in predictions}
    
    y_true = []
    y_pred = []
    
    for _, row in gt_df.iterrows():
        gt_country = str(row['Countries'])
        if gt_country in pred_dict:
            y_true.append(row['Inflation, 2022'])
            y_pred.append(pred_dict[gt_country])
    
    assert len(y_true) > 0, "No matching predictions found"
    
    print(f"Loaded {len(y_true)} ground truth and prediction pairs")
    
    # Calculate R²
    mean_actual = sum(y_true) / len(y_true)
    ss_tot = sum((a - mean_actual) ** 2 for a in y_true)
    ss_res = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Secondary threshold: R² ≥ 0.4 (moderate correlation, lenient)
    threshold_r2 = 0.4
    
    # Print metrics to logs
    print(f"\n=== R² Metrics ===")
    print(f"R² Score: {r2:.4f}")
    print(f"R² Threshold: {threshold_r2:.4f}")
    print(f"R² Status: {'PASS' if r2 >= threshold_r2 else 'FAIL'}")
    print(f"Number of samples: {len(y_true)}")
    print(f"Mean actual value: {mean_actual:.4f}")
    print(f"SS Total: {ss_tot:.4f}")
    print(f"SS Residual: {ss_res:.4f}")
    
    assert r2 >= threshold_r2, (
        f"R² {r2:.4f} is below secondary threshold {threshold_r2:.4f}"
    )

