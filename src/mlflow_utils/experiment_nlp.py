"""
experiment_nlp.py
──────────────────
MLflow experiment for the NLP pipeline (TF-IDF baseline + DistilBERT fine-tuning).

Logs:
  - Model type (tfidf / distilbert) and tokenizer/vectorizer config
  - Sentiment classification metrics (accuracy, macro-F1, per-class F1)
  - Rating regression MAE
  - Trained models / vectorizers as artifacts
  - NLP evaluation report JSON
"""

import sys
import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from mlflow_utils.mlflow_config import (
    EXPERIMENT_NLP, MODEL_REGISTRY_DISTILBERT, MLFLOW_TRACKING_URI
)
from mlflow_utils.mlflow_tracker import MLflowTracker


def run_nlp_experiment(mode: str = "tfidf"):
    """
    Run an NLP experiment run for the given mode.

    Args:
        mode: "tfidf" or "distilbert"
    """
    print("\n" + "=" * 60)
    print(f"  MLflow Experiment: NLP Sentiment ({mode.upper()})")
    print("=" * 60)

    from nlp.nlp_config import (
        NLP_EVALUATION_REPORT, TFIDF_MODEL_PATH,
        MAX_TOKEN_LENGTH, BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LEARNING_RATE,
    )
    from nlp.nlp_data_prep import NLPDataPrep
    from nlp.nlp_trainer   import NLPTrainer
    from nlp.nlp_evaluator import NLPEvaluator

    # Data prep
    data_prep = NLPDataPrep()
    data_dict = data_prep.prepare()
    train_df  = data_dict["train_df"]
    test_df   = data_dict["test_df"]

    # Train & Evaluate
    trainer   = NLPTrainer()
    evaluator = NLPEvaluator()

    if mode == "tfidf":
        model_obj = trainer.train_tfidf(train_df)
        metrics   = evaluator.evaluate_tfidf(test_df)
    else:
        model_obj = trainer.train_distilbert(train_df, test_df)
        metrics   = evaluator.evaluate_distilbert(test_df)

    # MLflow run
    tracker = MLflowTracker(EXPERIMENT_NLP, tags={"module": "nlp", "model_type": mode})
    tracker.start(run_name=f"{mode.upper()}-sentiment")

    try:
        common_params = {
            "model_type":   mode,
            "dataset":      "Amazon-Consumer-Electronics-Reviews",
            "task":         "sentiment_classification",
            "num_classes":  3,
            "classes":      "Negative,Neutral,Positive",
        }
        if mode == "tfidf":
            common_params.update({
                "vectorizer":   "TF-IDF",
                "max_features": 10000,
                "ngram_range":  "(1,2)",
                "classifier":   "LogisticRegression",
            })
        else:
            common_params.update({
                "base_model":    "distilbert-base-uncased",
                "max_len":       MAX_TOKEN_LENGTH,
                "batch_size":    BERT_BATCH_SIZE,
                "epochs":        BERT_EPOCHS,
                "learning_rate": BERT_LEARNING_RATE,
                "optimizer":     "AdamW",
            })
        tracker.log_params(common_params)

        if metrics:
            tracker.log_metrics({
                "accuracy":  float(metrics.get("accuracy",  0)),
                "macro_f1":  float(metrics.get("macro_f1",  0)),
            })

        # Artifacts
        if os.path.exists(NLP_EVALUATION_REPORT):
            tracker.log_artifact(NLP_EVALUATION_REPORT, "reports")

        if mode == "tfidf" and os.path.exists(TFIDF_MODEL_PATH):
            tracker.log_artifact(TFIDF_MODEL_PATH, "models")

        if mode == "distilbert":
            from nlp.distilbert_model import DistilBertSentimentModel
            if isinstance(model_obj, DistilBertSentimentModel):
                tracker.log_model_pytorch(
                    model_obj,
                    "distilbert_sentiment",
                    registered_name=MODEL_REGISTRY_DISTILBERT,
                )

        print(f"  ✅ NLP ({mode}) run logged.")
    finally:
        tracker.end()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
    run_nlp_experiment(mode)
