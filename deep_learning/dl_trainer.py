import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from src.deep_learning.dl_config import EPOCHS, LEARNING_RATE, PATIENCE, DEVICE

class DLTrainer:
    def __init__(self, model: nn.Module, model_save_path: str):
        self.model = model.to(DEVICE)
        self.model_save_path = model_save_path
        # Weighted BCE Loss to handle severe class imbalance (~3.4% failure positive rate)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(y_batch)

            train_loss /= len(train_loader.dataset)

            # Validation
            val_loss, val_f1 = self.evaluate(val_loader)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Best Validation F1: {best_val_f1:.4f}")
        return best_val_f1

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        return avg_loss, f1
