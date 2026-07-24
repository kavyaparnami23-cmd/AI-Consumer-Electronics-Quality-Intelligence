"""
nlp/nlp_evaluator.py -- Evaluation component for NLP models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from logger import logger
from exception import CustomException
from nlp.nlp_config import (
    NLP_EVALUATION_REPORT,
    NLP_PLOTS_DIR,
    BERT_BATCH_SIZE,
    SENTIMENT_LABELS,
)
from nlp.tfidf_model import TfidfSentimentModel
from nlp.utils import sentiment_to_id


class NLPEvaluator:
    """
    Evaluates trained NLP models on test dataset and generates reports & plots.
    """

    def __init__(self):
        self.report_path = NLP_EVALUATION_REPORT
        self.plots_dir = NLP_PLOTS_DIR

    def _plot_confusion_matrix(self, cm, title: str, save_path: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=SENTIMENT_LABELS,
            yticklabels=SENTIMENT_LABELS,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def evaluate_tfidf(self, test_df) -> dict:
        """
        Evaluate TF-IDF baseline model.
        """
        try:
            logger.info("Evaluating TF-IDF Model...")
            model = TfidfSentimentModel()
            model.load()

            texts = test_df["clean_text"].tolist()
            y_true = test_df["sentiment"].tolist()

            res = model.predict(texts)
            y_pred = res["predictions"]

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
            clf_rep = classification_report(y_true, y_pred, target_names=SENTIMENT_LABELS)
            cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)

            cm_path = os.path.join(self.plots_dir, "tfidf_confusion_matrix.png")
            self._plot_confusion_matrix(cm, "TF-IDF Confusion Matrix", cm_path)

            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm.tolist(),
                "classification_report": clf_rep,
            }
            logger.info(f"TF-IDF Evaluation Results: {metrics['f1_score']}")
            return metrics

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

    def evaluate_distilbert(self, test_df) -> dict:
        """
        Evaluate fine-tuned DistilBERT model.
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from nlp.dataset import SentimentDataset
            from nlp.distilbert_model import DistilBERTSentimentModel

            logger.info("Evaluating DistilBERT Model...")
            model = DistilBERTSentimentModel(pretrained=False)
            model.load()

            texts = test_df["clean_text"].tolist()
            y_true_str = test_df["sentiment"].tolist()
            y_true_ids = [sentiment_to_id(s) for s in y_true_str]

            test_dataset = SentimentDataset(
                texts=texts,
                labels=y_true_ids,
                tokenizer=model.tokenizer,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BERT_BATCH_SIZE,
                shuffle=False,
            )

            all_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    preds, _ = model.predict(input_ids, attention_mask)
                    all_preds.extend(preds.cpu().numpy().tolist())

            y_pred_str = [SENTIMENT_LABELS[i] for i in all_preds]

            acc = accuracy_score(y_true_str, y_pred_str)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true_str, y_pred_str, average="weighted")
            clf_rep = classification_report(y_true_str, y_pred_str, target_names=SENTIMENT_LABELS)
            cm = confusion_matrix(y_true_str, y_pred_str, labels=SENTIMENT_LABELS)

            cm_path = os.path.join(self.plots_dir, "distilbert_confusion_matrix.png")
            self._plot_confusion_matrix(cm, "DistilBERT Confusion Matrix", cm_path)

            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm.tolist(),
                "classification_report": clf_rep,
            }
            logger.info(f"DistilBERT Evaluation Results: {metrics['f1_score']}")
            return metrics

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

    def evaluate_all(self, test_df) -> dict:
        """
        Evaluate both models and save evaluation report.
        """
        try:
            tfidf_metrics = self.evaluate_tfidf(test_df)
            try:
                distilbert_metrics = self.evaluate_distilbert(test_df)
                best_model = "distilbert" if distilbert_metrics["f1_score"] >= tfidf_metrics["f1_score"] else "tfidf"
            except Exception as e:
                logger.warning(f"DistilBERT evaluation skipped or failed: {e}")
                distilbert_metrics = None
                best_model = "tfidf"

            report = {
                "tfidf_baseline": tfidf_metrics,
                "distilbert": distilbert_metrics,
                "best_model": best_model,
            }

            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=4)

            print("\n" + "=" * 60)
            print("  NLP EVALUATION SUMMARY")
            print("=" * 60)
            print(f"  TF-IDF Baseline F1 : {tfidf_metrics['f1_score']}")
            if distilbert_metrics:
                print(f"  DistilBERT F1      : {distilbert_metrics['f1_score']}")
            print(f"  Winner             : {report['best_model'].upper()}")
            print("=" * 60)

            return report

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    import pandas as pd
    sample_df = pd.DataFrame({
        "clean_text": ["Great product", "Horrible battery"],
        "sentiment": ["Positive", "Negative"]
    })
    evaluator = NLPEvaluator()
    evaluator.evaluate_tfidf(sample_df)
