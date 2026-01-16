import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# Input parameters

TRAIN_SPLIT = 0.8  # Use 80% for training, 20% for testing
ID_COLUMN = "id"
# Target columns for multi-label classification (last 7 columns are the fault types)
# TARGET_COLUMNS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
# For task_3, use single Target column
TARGET_COLUMNS = ["Target"]

# ------------------------------------------------------------

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Task root points to task_3 directory
TASK_ROOT = os.path.join(SCRIPT_DIR, "tasks", "task_3")
# Construct path to data/train.csv
TRAIN_CSV_PATH = os.path.join(TASK_ROOT, "data", "train.csv")

df = pd.read_csv(TRAIN_CSV_PATH)
original_id = df[ID_COLUMN].copy()
del df[ID_COLUMN]

# Shuffle the data with fixed random seed
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split point
n_total = len(df_shuffled)
n_train = int(n_total * TRAIN_SPLIT)

train = df_shuffled[:n_train].reset_index(drop=True)
test = df_shuffled[n_train:].reset_index(drop=True)

# Add id column to both train and test
train.insert(loc=0, column=ID_COLUMN, value=train.index)
test.insert(loc=0, column=ID_COLUMN, value=test.index)

# Create submission CSV with all target columns
submission = test[[ID_COLUMN] + TARGET_COLUMNS].copy()
submission_csv_path = os.path.join(TASK_ROOT, "tests", "test_ground_truth.csv")
# Create tests directory if it doesn't exist
os.makedirs(os.path.dirname(submission_csv_path), exist_ok=True)
submission.to_csv(submission_csv_path, index=False)

# Drop all target columns from test
test = test.drop(columns=TARGET_COLUMNS)

# Save train into /data folder
train.to_csv(TRAIN_CSV_PATH, index=False)

# Save test into /tests folder
tests_dir = os.path.join(TASK_ROOT, "tests")
os.makedirs(tests_dir, exist_ok=True)
test_csv_path = os.path.join(tests_dir, "test.csv")
test.to_csv(test_csv_path, index=False)

print(f"Successfully split data:")
print(f"  Training samples: {len(train)} ({100*len(train)/n_total:.1f}%)")
print(f"  Test samples: {len(test)} ({100*len(test)/n_total:.1f}%)")
print(f"  Train saved to: {TRAIN_CSV_PATH}")
print(f"  Test saved to: {test_csv_path}")
print(f"  Test ground truth saved to: {submission_csv_path}")
