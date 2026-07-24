"""
nlp/nlp_trainer.py -- Trainer class for DistilBERT and TF-IDF models.
"""

import sys

from logger import logger
from exception import CustomException
from nlp.nlp_config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_WARMUP_STEPS,
    BERT_WEIGHT_DECAY,
    BERT_EARLY_STOPPING,
    DEVICE,
)
from nlp.tfidf_model import TfidfSentimentModel
from nlp.utils import sentiment_to_id


class NLPTrainer:
    """
    Handles training/fine-tuning of NLP models (TF-IDF Baseline & DistilBERT).
    """

    def __init__(self):
        pass

    def train_tfidf(self, train_df) -> TfidfSentimentModel:
        """
        Train the TF-IDF + Logistic Regression baseline model.
        """
        try:
            logger.info("Training TF-IDF Baseline Model...")
            print("\nTraining TF-IDF Baseline Model...")

            model = TfidfSentimentModel()
            train_texts = train_df["clean_text"].tolist()
            train_labels = train_df["sentiment"].tolist()

            metrics = model.train(train_texts, train_labels)
            model.save()

            print(f"TF-IDF Training Completed. Train F1: {metrics['train_f1']}")
            return model

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

    def train_distilbert(self, train_df, val_df=None):
        """
        Fine-tune DistilBERT sequence classification model.
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup
            from nlp.dataset import SentimentDataset
            from nlp.distilbert_model import DistilBERTSentimentModel

            logger.info("Fine-tuning DistilBERT Model...")
            print("\nFine-tuning DistilBERT Model...")

            model = DistilBERTSentimentModel(pretrained=True)

            train_texts = train_df["clean_text"].tolist()
            train_labels = [sentiment_to_id(s) for s in train_df["sentiment"]]

            train_dataset = SentimentDataset(
                texts=train_texts,
                labels=train_labels,
                tokenizer=model.tokenizer,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=BERT_BATCH_SIZE,
                shuffle=True,
            )

            if val_df is not None:
                val_texts = val_df["clean_text"].tolist()
                val_labels = [sentiment_to_id(s) for s in val_df["sentiment"]]
                val_dataset = SentimentDataset(
                    texts=val_texts,
                    labels=val_labels,
                    tokenizer=model.tokenizer,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=BERT_BATCH_SIZE,
                    shuffle=False,
                )
            else:
                val_loader = None

            optimizer = AdamW(
                model.model.parameters(),
                lr=BERT_LEARNING_RATE,
                weight_decay=BERT_WEIGHT_DECAY,
            )

            total_steps = len(train_loader) * BERT_EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=BERT_WARMUP_STEPS,
                num_training_steps=total_steps,
            )

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(BERT_EPOCHS):
                model.model.train()
                total_train_loss = 0.0

                print(f"  Epoch {epoch + 1}/{BERT_EPOCHS} ...", end="", flush=True)

                for step, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    outputs = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    loss = outputs.loss
                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                avg_train_loss = total_train_loss / len(train_loader)
                logger.info(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

                if val_loader is not None:
                    model.model.eval()
                    total_val_loss = 0.0

                    with torch.no_grad():
                        for batch in val_loader:
                            input_ids = batch["input_ids"]
                            attention_mask = batch["attention_mask"]
                            labels = batch["labels"]

                            outputs = model.forward(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                            total_val_loss += outputs.loss.item()

                    avg_val_loss = total_val_loss / len(val_loader)
                    print(f" Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        model.save()
                    else:
                        patience_counter += 1
                        if patience_counter >= BERT_EARLY_STOPPING:
                            print("\nEarly stopping triggered.")
                            logger.info("Early stopping triggered during DistilBERT fine-tuning.")
                            break
                else:
                    print(f" Train Loss: {avg_train_loss:.4f}")
                    model.save()

            print("DistilBERT Fine-tuning Completed.")
            return model

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    import pandas as pd
    sample_df = pd.DataFrame({
        "clean_text": ["Great battery life", "Broke after one day", "Average product"],
        "sentiment": ["Positive", "Negative", "Neutral"]
    })
    trainer = NLPTrainer()
    trainer.train_tfidf(sample_df)
