"""
nlp/distilbert_model.py -- DistilBERT-based sentiment classification model.

Wraps HuggingFace DistilBertForSequenceClassification for 3-class
sentiment prediction (Negative / Neutral / Positive).
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

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

    Architecture
    ------------
    DistilBERT base (66M params, 6 transformer layers)
        -> [CLS] token representation (768-dim)
        -> Linear(768, 3) classification head
        -> Softmax -> {Negative, Neutral, Positive}

    Why DistilBERT instead of full BERT:
    - 40% smaller, 60% faster
    - Retains 97% of BERT's accuracy
    - Practical for CPU training
    """

    def __init__(self, pretrained: bool = True):
        """
        Parameters
        ----------
        pretrained : bool
            If True, load from HuggingFace hub.
            If False, expect to call self.load() with local weights.
        """
        try:
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

            self.device = torch.device(DEVICE)

            if self.model is not None:
                self.model.to(self.device)

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Forward pass (used during training)
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Run a forward pass through the model.

        Returns HuggingFace output object with .loss and .logits
        """
        return self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        """
        Returns
        -------
        predictions  : Tensor of predicted class IDs
        probabilities: Tensor of softmax probabilities
        """
        self.model.eval()
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self):
        """Save model weights and tokenizer to local directories."""
        os.makedirs(DISTILBERT_MODEL_DIR, exist_ok=True)
        os.makedirs(DISTILBERT_TOKENIZER_DIR, exist_ok=True)

        self.model.save_pretrained(DISTILBERT_MODEL_DIR)
        self.tokenizer.save_pretrained(DISTILBERT_TOKENIZER_DIR)

        logger.info(f"DistilBERT model saved to     : {DISTILBERT_MODEL_DIR}")
        logger.info(f"DistilBERT tokenizer saved to  : {DISTILBERT_TOKENIZER_DIR}")
        print(f"  DistilBERT model saved to : {DISTILBERT_MODEL_DIR}")

    def load(self):
        """Load model weights and tokenizer from local directories."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_MODEL_DIR,
            num_labels=NUM_CLASSES,
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_TOKENIZER_DIR)
        self.model.to(self.device)
        self.model.eval()

        logger.info("DistilBERT model loaded from disk")

    # ------------------------------------------------------------------
    # Freeze / Unfreeze (for transfer learning control)
    # ------------------------------------------------------------------

    def freeze_base(self):
        """Freeze all DistilBERT layers except the classification head."""
        for param in self.model.distilbert.parameters():
            param.requires_grad = False
        logger.info("DistilBERT base layers frozen (only classifier is trainable)")

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All DistilBERT parameters unfrozen")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Count trainable vs total parameters."""
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params":     total,
            "trainable_params": trainable,
            "frozen_params":    total - trainable,
        }


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DistilBERT Model Loading")
    print("=" * 60)

    model = DistilBERTSentimentModel(pretrained=True)
    params = model.count_parameters()
    print(f"Total params     : {params['total_params']:,}")
    print(f"Trainable params : {params['trainable_params']:,}")

    # Quick test inference
    tokens = model.tokenizer(
        "This product is fantastic!",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    preds, probs = model.predict(tokens["input_ids"], tokens["attention_mask"])
    print(f"Prediction : {preds.item()} | Probabilities : {probs.squeeze().tolist()}")
    print("=" * 60)
