"""
nlp/utils.py -- NLP-specific utility functions.

Text cleaning, label encoding/decoding, object persistence.
"""

import os
import re
import html
import string
import joblib

from nlp.nlp_config import SENTIMENT_LABELS


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean a single review text string:
      1. Unescape HTML entities
      2. Remove HTML tags
      3. Lowercase
      4. Remove URLs
      5. Remove digits
      6. Remove excessive punctuation
      7. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)                          # &amp; -> &
    text = re.sub(r"<[^>]+>", " ", text)                 # strip HTML tags
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\d+", " ", text)                      # digits
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)  # punct
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


# ============================================================
# Label Encoding / Decoding
# ============================================================

def sentiment_to_id(label: str) -> int:
    """Convert sentiment label string to integer ID."""
    return SENTIMENT_LABELS.index(label)


def id_to_sentiment(idx: int) -> str:
    """Convert integer ID back to sentiment label string."""
    return SENTIMENT_LABELS[idx]


# ============================================================
# Object Persistence
# ============================================================

def save_nlp_object(file_path: str, obj) -> None:
    """Save an NLP artifact (model, vectorizer, encoder) to disk."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(obj, file_path)


def load_nlp_object(file_path: str):
    """Load an NLP artifact from disk."""
    return joblib.load(file_path)


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    sample = "<p>This battery is AMAZING!! Check https://amzn.com/123 for 50% off.</p>"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
    print()
    print("Positive ->", sentiment_to_id("Positive"))
    print("ID 0     ->", id_to_sentiment(0))
