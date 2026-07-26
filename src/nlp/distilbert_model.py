"""
nlp/distilbert_model.py -- DistilBERT-based sentiment classification model.

Wraps HuggingFace DistilBertForSequenceClassification for 3-class
sentiment prediction (Negative / Neutral / Positive).
"""

import os
import sys

from logger import logger
from exception import CustomException
from nlp.nlp_config import (
    DISTILBERT_MODEL_NAME,
    DISTILBERT_MODEL_DIR,
    DISTILBERT_TOKENIZER_DIR,
    NUM_CLASSES,
    DEVICE,
)


class DistilBERTSentimentModel:
    """
    Fine-tuned DistilBERT for 3-class sentiment classification.
    """

    def __init__(self, pretrained: bool = True):
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            )

            self.torch = torch
            self.device = torch.device(DEVICE)

            if pretrained:
                logger.info(f"Loading DistilBERT from HuggingFace: {DISTILBERT_MODEL_NAME}")
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    DISTILBERT_MODEL_NAME,
                    num_labels=NUM_CLASSES,
                )
                self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
            else:
                self.model     = None
                self.tokenizer = None

            if self.model is not None:
                self.model.to(self.device)

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )

    def predict(self, input_ids, attention_mask):
        import torch
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def save(self):
        os.makedirs(DISTILBERT_MODEL_DIR, exist_ok=True)
        os.makedirs(DISTILBERT_TOKENIZER_DIR, exist_ok=True)

        self.model.save_pretrained(DISTILBERT_MODEL_DIR)
        self.tokenizer.save_pretrained(DISTILBERT_TOKENIZER_DIR)

        logger.info(f"DistilBERT model saved to     : {DISTILBERT_MODEL_DIR}")
        logger.info(f"DistilBERT tokenizer saved to  : {DISTILBERT_TOKENIZER_DIR}")
        print(f"  DistilBERT model saved to : {DISTILBERT_MODEL_DIR}")

    def load(self):
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizer,
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_MODEL_DIR,
            num_labels=NUM_CLASSES,
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_TOKENIZER_DIR)
        self.model.to(self.device)
        self.model.eval()

        logger.info("DistilBERT model loaded from disk")

    def freeze_base(self):
        for param in self.model.distilbert.parameters():
            param.requires_grad = False
        logger.info("DistilBERT base layers frozen (only classifier is trainable)")

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All DistilBERT parameters unfrozen")

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params":     total,
            "trainable_params": trainable,
            "frozen_params":    total - trainable,
        }
