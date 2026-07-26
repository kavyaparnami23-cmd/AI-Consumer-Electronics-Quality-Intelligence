"""
run_all_experiments.py
───────────────────────
Master runner — executes all 4 MLflow experiments in sequence:

  1. Classic ML   (XGBoost / LightGBM / RandomForest)
  2. Deep Learning (BiLSTM + 1D-CNN)
  3. NLP TF-IDF   (TF-IDF + Logistic Regression)
  4. Time Series  (LSTM Autoencoder + Isolation Forest)

After all runs, prints a summary table with run IDs and key metrics.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --skip-classic --skip-nlp   # selective
"""

import sys
import os
import argparse
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import mlflow
from src.mlflow_utils.mlflow_config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_CLASSIC_ML, EXPERIMENT_DEEP_LEARNING,
    EXPERIMENT_NLP, EXPERIMENT_TIME_SERIES,
)


def _header(title: str):
    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║  {title:<56}║")
    print("╚" + "═" * 58 + "╝")


def _ensure_run_ended():
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


def run_all(args):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    results = []

    # ── 1. Classic ML ─────────────────────────────────────────────────────────
    if not args.skip_classic:
        _ensure_run_ended()
        _header("Experiment 1/4: Classic ML Sensor Failure")
        try:
            from src.mlflow_utils.experiment_classic_ml import run_classic_ml_experiment
            t0 = time.time()
            run_classic_ml_experiment()
            results.append(("Classic ML", "✅", f"{time.time()-t0:.0f}s"))
        except Exception as e:
            print(f"  ❌ Classic ML failed: {e}")
            results.append(("Classic ML", "❌", str(e)[:40]))
        finally:
            _ensure_run_ended()

    # ── 2. Deep Learning ──────────────────────────────────────────────────────
    if not args.skip_dl:
        _ensure_run_ended()
        _header("Experiment 2/4: Deep Learning (BiLSTM & 1D-CNN)")
        try:
            from src.mlflow_utils.experiment_deep_learning import run_dl_experiment
            t0 = time.time()
            run_dl_experiment()
            results.append(("Deep Learning", "✅", f"{time.time()-t0:.0f}s"))
        except Exception as e:
            print(f"  ❌ Deep Learning failed: {e}")
            results.append(("Deep Learning", "❌", str(e)[:40]))
        finally:
            _ensure_run_ended()

    # ── 3. NLP ────────────────────────────────────────────────────────────────
    if not args.skip_nlp:
        _ensure_run_ended()
        _header("Experiment 3/4: NLP Sentiment (TF-IDF)")
        try:
            from src.mlflow_utils.experiment_nlp import run_nlp_experiment
            t0 = time.time()
            run_nlp_experiment(mode="tfidf")
            results.append(("NLP TF-IDF", "✅", f"{time.time()-t0:.0f}s"))
        except Exception as e:
            print(f"  ❌ NLP failed: {e}")
            results.append(("NLP TF-IDF", "❌", str(e)[:40]))
        finally:
            _ensure_run_ended()

    # ── 4. Time Series ────────────────────────────────────────────────────────
    if not args.skip_ts:
        _ensure_run_ended()
        _header("Experiment 4/4: Time Series Anomaly Detection")
        try:
            from src.mlflow_utils.experiment_time_series import run_ts_experiment
            t0 = time.time()
            run_ts_experiment()
            results.append(("Time Series", "✅", f"{time.time()-t0:.0f}s"))
        except Exception as e:
            print(f"  ❌ Time Series failed: {e}")
            results.append(("Time Series", "❌", str(e)[:40]))
        finally:
            _ensure_run_ended()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<25} {'Status':<8} {'Duration/Note'}")
    print("  " + "-" * 55)
    for name, status, note in results:
        print(f"  {name:<25} {status:<8} {note}")
    print("=" * 60)
    print(f"\n  📊 View results:  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"     Then open:      http://localhost:5000\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all MLflow experiments")
    parser.add_argument("--skip-classic", action="store_true")
    parser.add_argument("--skip-dl",      action="store_true")
    parser.add_argument("--skip-nlp",     action="store_true")
    parser.add_argument("--skip-ts",      action="store_true")
    args = parser.parse_args()
    run_all(args)
