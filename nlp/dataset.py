"""
nlp/dataset.py -- PyTorch Dataset for DistilBERT sentiment classification.
"""

from nlp.nlp_config import (
    DISTILBERT_MODEL_NAME,
    MAX_TOKEN_LENGTH,
    SENTIMENT_LABELS,
)

try:
    import torch
    from torch.utils.data import Dataset
    BaseDataset = Dataset
except ImportError:
    BaseDataset = object
    torch = None


class SentimentDataset(BaseDataset):
    """
    PyTorch Dataset that tokenises review text using the
    DistilBERT tokenizer and returns tensors ready for the model.
    """

    def __init__(
        self,
        texts: list,
        labels: list = None,
        tokenizer=None,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        if torch is None:
            raise ImportError("PyTorch (torch) is required to use SentimentDataset.")
        from transformers import DistilBertTokenizer

        self.texts      = texts
        self.labels     = labels
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


if __name__ == "__main__":
    if torch is not None:
        sample_texts  = ["This product is amazing!", "Terrible quality, broke on day 1"]
        sample_labels = [2, 0]   # Positive, Negative

        ds = SentimentDataset(sample_texts, sample_labels)
        item = ds[0]

        print("input_ids shape     :", item["input_ids"].shape)
        print("attention_mask shape:", item["attention_mask"].shape)
        print("label               :", item["labels"].item())
        print("Decoded tokens      :", ds.tokenizer.decode(item["input_ids"]))
