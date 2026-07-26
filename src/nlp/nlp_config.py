import os

# Set HuggingFace cache directory to project directory (avoids C: drive space limit)
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "huggingface"))


# ---- Paths ----
NLP_DATASET_PATH     = os.path.join("datasets", "clean_reviews.csv")
NLP_ARTIFACT_DIR     = os.path.join("artifacts", "nlp")
NLP_PLOTS_DIR        = os.path.join(NLP_ARTIFACT_DIR, "plots")

NLP_PROCESSED_TRAIN  = os.path.join(NLP_ARTIFACT_DIR, "nlp_train.csv")
NLP_PROCESSED_TEST   = os.path.join(NLP_ARTIFACT_DIR, "nlp_test.csv")

TFIDF_MODEL_PATH        = os.path.join("saved_models", "nlp", "tfidf_model.pkl")
TFIDF_VECTORIZER_PATH   = os.path.join("saved_models", "nlp", "tfidf_vectorizer.pkl")
TFIDF_LABEL_ENCODER_PATH = os.path.join("saved_models", "nlp", "tfidf_label_encoder.pkl")

DISTILBERT_MODEL_DIR     = os.path.join("saved_models", "nlp", "distilbert")
DISTILBERT_TOKENIZER_DIR = os.path.join("saved_models", "nlp", "distilbert_tokenizer")

NLP_EVALUATION_REPORT    = os.path.join(NLP_ARTIFACT_DIR, "nlp_evaluation_report.json")

# ---- Text Column Names ----
TEXT_COLUMN     = "reviews.text"
TITLE_COLUMN    = "reviews.title"
RATING_COLUMN   = "reviews.rating"
SENTIMENT_COLUMN = "sentiment"

# ---- Sentiment Mapping ----
# 1,2 -> Negative | 3 -> Neutral | 4,5 -> Positive
SENTIMENT_MAP = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive",
}
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
NUM_CLASSES = len(SENTIMENT_LABELS)

# ---- Data Split ----
NLP_TEST_SIZE    = 0.20
NLP_RANDOM_STATE = 42

# ---- TF-IDF Hyperparameters ----
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE  = (1, 2)      # unigrams + bigrams

# ---- DistilBERT Hyperparameters ----
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH      = 128       # max tokens per review
BERT_BATCH_SIZE       = 16        # small for CPU
BERT_EPOCHS           = 3
BERT_LEARNING_RATE    = 2e-5
BERT_WARMUP_STEPS     = 0
BERT_WEIGHT_DECAY     = 0.01
BERT_EARLY_STOPPING   = 2        # patience in epochs

# ---- Device ----
DEVICE = "cpu"
