# postprocess_all.py

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score

RESULTS_ROOT = "results"
OUT_PATH = os.path.join(RESULTS_ROOT, "final_metrics.json")

# 1) Collect ALL inner-fold validation logits for threshold calculation
all_thresholds = {0: [], 2: []}  # Class 0 vs rest, Class 2 vs rest

for fold_test in range(5):
    for fold_inner in range(5):
        val_path = os.path.join(RESULTS_ROOT, f"test_{fold_test}_val_{fold_inner}.csv")
        if os.path.exists(val_path):
            df_val = pd.read_csv(val_path)
            y_true = df_val["True"].values

            # Malignant vs Benign (class 0 is benign, so positive = y != 0)
            y_true_m = (y_true != 0).astype(int)
            y_prob_m = 1 - df_val["prob0"].values  # P(malignant)
            fpr, tpr, thr = roc_curve(y_true_m, y_prob_m)
            best_thr = thr[np.argmax(tpr - fpr)]
            all_thresholds[0].append(best_thr)

            # Aggressive vs Indolent (positive = class 2)
            y_true_a = (y_true == 2).astype(int)
            y_prob_a = df_val["prob2"].values
            fpr, tpr, thr = roc_curve(y_true_a, y_prob_a)
            best_thr = thr[np.argmax(tpr - fpr)]
            all_thresholds[2].append(best_thr)

# 2) Compute ONE GLOBAL THRESHOLD per class
global_thresholds = {
    0: float(np.mean(all_thresholds[0])) if all_thresholds[0] else 0.5,
    2: float(np.mean(all_thresholds[2])) if all_thresholds[2] else 0.5
}

# 3) Concatenate ALL outer test logits
all_test = []
for fold_test in range(5):
    test_path = os.path.join(RESULTS_ROOT, f"test_{fold_test}.csv")
    if os.path.exists(test_path):
        all_test.append(pd.read_csv(test_path))

if not all_test:
    raise RuntimeError("No test_?.csv files found.")

test_df = pd.concat(all_test, ignore_index=True)

# 4) Compute final metrics ONCE
def compute_final_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "AUC": float(auc),
        "Threshold": float(thr),
        "Sensitivity": float(tp / (tp + fn) if (tp + fn) else 0.0),
        "Specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ConfusionMatrix": cm.tolist()
    }

results = {}

# Malignant vs Benign (class 0 vs rest; positive = malignant)
y_true_m = (test_df["True"].values != 0).astype(int)
y_prob_m = 1 - test_df["prob0"].values
results["class_0_vs_rest"] = compute_final_metrics(y_true_m, y_prob_m, global_thresholds[0])

# Aggressive vs Indolent (class 2 vs rest; positive = aggressive)
y_true_a = (test_df["True"].values == 2).astype(int)
y_prob_a = test_df["prob2"].values
results["class_2_vs_rest"] = compute_final_metrics(y_true_a, y_prob_a, global_thresholds[2])

# Save JSON
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Final metrics saved to: {OUT_PATH}")
print("✅ Global thresholds:", global_thresholds)
