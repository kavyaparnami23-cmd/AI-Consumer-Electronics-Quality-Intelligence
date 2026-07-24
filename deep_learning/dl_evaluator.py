import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.deep_learning.dl_config import DEVICE, REPORT_PATH

class DLEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, model, data_loader, model_name: str):
        model.eval()
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        clf_rep = classification_report(all_targets, all_preds, zero_division=0)

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "classification_report": clf_rep
        }

        self.results[model_name] = metrics
        return metrics

    def save_report(self):
        with open(REPORT_PATH, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"DL Evaluation Report saved to: {REPORT_PATH}")
