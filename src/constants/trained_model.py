"""Constants for the trained news-impact model and the filter built on it."""

from pathlib import Path

# --- Artifact location --------------------------------------------------------

# Directory the trained model artifact is written to and loaded from.
TRAINED_MODEL_DIR = Path("trained_models")

# Full path of the serialized scikit-learn pipeline (vectorizer + network).
TRAINED_MODEL_PATH = TRAINED_MODEL_DIR.joinpath("news_impact_mlp.joblib")

# --- TF-IDF text features -----------------------------------------------------

# Vocabulary cap, word n-gram range and minimum document frequency for the
# TF-IDF vectorizer that turns a headline into a sparse feature vector.
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# --- Dimensionality reduction -------------------------------------------------

# Number of latent-semantic components the sparse TF-IDF vectors are reduced to
# (a dense, denoised input the network trains on far more stably).
SVD_COMPONENTS = 300

# --- Neural network (multilayer perceptron) -----------------------------------

# Hidden-layer sizes, L2 regularization strength and training cap for the MLP
# regressor that predicts a headline's abnormal price move.
MLP_HIDDEN_LAYERS = (256, 64)
MLP_ALPHA = 1e-3
MLP_MAX_ITER = 300

# --- Training reproducibility -------------------------------------------------

# Seed shared by the randomized SVD, the network and the row-loading order so a
# rebuild is byte-for-byte reproducible.
TRAIN_SEED = 42

# Fraction of the (time-ordered) training rows held out to report honest
# validation metrics before the final model is refit on everything.
VALIDATION_FRACTION = 0.15
