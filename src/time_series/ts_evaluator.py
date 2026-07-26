"""
ts_evaluator.py
───────────────
Evaluates the anomaly detection system against ground-truth Machine failure labels.

Metrics reported:
  - Precision, Recall, F1 (macro & per-class)
  - ROC-AUC (anomaly_score vs label)
  - Average Precision (AP)  — better suited for imbalanced datasets
  - Confusion matrix values (TP, FP, TN, FN)

Saves:
  - artifacts/time_series/ts_anomaly_report.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from src.time_series.ts_config import TS_REPORT_PATH


def evaluate_anomaly_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """
    Compute full evaluation metrics for the anomaly detector.

    Args:
        y_true:        Ground-truth Machine failure labels (0/1).
        y_pred:        Binary combined_flag predictions (0/1).
        anomaly_scores: Continuous anomaly score ∈ [0, 1].

    Returns:
        Metrics dict (also saved to JSON).
    """
    # ── Classification metrics ─────────────────────────────────────────────────
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(   y_true, y_pred, zero_division=0)
    f1        = f1_score(       y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        roc_auc = 0.0

    try:
        avg_prec = average_precision_score(y_true, anomaly_scores)
    except ValueError:
        avg_prec = 0.0

    # ── Confusion matrix ───────────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cls_report = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )

    # ── Summary stats ──────────────────────────────────────────────────────────
    n_anomalies   = int(y_pred.sum())
    n_true_fail   = int(y_true.sum())
    n_samples     = len(y_true)
    detected_rate = round(tp / max(n_true_fail, 1) * 100, 2)   # % of true failures caught

    metrics = {
        "n_samples":           n_samples,
        "n_true_failures":     n_true_fail,
        "n_detected_anomalies": n_anomalies,
        "true_positives":      int(tp),
        "false_positives":     int(fp),
        "true_negatives":      int(tn),
        "false_negatives":     int(fn),
        "detection_rate_pct":  detected_rate,
        "precision":           round(precision, 4),
        "recall":              round(recall,    4),
        "f1_score":            round(f1,        4),
        "roc_auc":             round(roc_auc,   4),
        "average_precision":   round(avg_prec,  4),
        "classification_report": cls_report,
    }

    # Persist report
    with open(TS_REPORT_PATH, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=4)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  TIME SERIES ANOMALY DETECTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples          : {n_samples:,}")
    print(f"  True Failures    : {n_true_fail:,}  ({n_true_fail/n_samples*100:.1f}%)")
    print(f"  Detected Anomalies: {n_anomalies:,}")
    print(f"  Detection Rate   : {detected_rate:.1f}% of true failures caught")
    print(f"  Precision : {precision:.4f}  |  Recall : {recall:.4f}  |  F1 : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}  |  Avg Precision : {avg_prec:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 60)
    print(cls_report)
    print(f"  Report saved → {TS_REPORT_PATH}")

    return metrics
