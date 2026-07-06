import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from src.constants.trained_model import (
    MLP_ALPHA,
    MLP_HIDDEN_LAYERS,
    MLP_MAX_ITER,
    SVD_COMPONENTS,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TRAIN_SEED,
    TRAINED_MODEL_DIR,
    TRAINED_MODEL_PATH,
    VALIDATION_FRACTION,
)
from src.dataset.news_sampler import NewsSampler
from src.evaluation.news_scorer import NewsScorer
from src.filtering.trained_model_filter import TrainedModelFilter
from src.models.sampled_news import SampledNews
from src.utils.enums.price_horizon import PriceHorizon
from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.joinpath("data")


def _row_count(dataset: Path) -> int:
    """Return the number of news rows in a dataset database."""
    connection = sqlite3.connect(str(dataset))
    try:
        return connection.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    finally:
        connection.close()


def _load_rows(dataset: Path) -> list[SampledNews]:
    """Load every news row of one dataset as SampledNews objects."""
    count = _row_count(dataset)
    if count == 0:
        return []
    sampler = NewsSampler(database=dataset, batch_size=count, min_gap_days=0, seed=TRAIN_SEED)
    return sampler.sample()


def _build_examples(rows: list[SampledNews]):
    """Turn rows into (texts, abnormal-1d-return targets, dates), dropping unlabeled ones."""
    scorer = NewsScorer(subtract_market=True)
    texts: list[str] = []
    targets: list[float] = []
    dates = []
    for row in rows:
        target = scorer.price_change(row, PriceHorizon.ONE_DAY)
        if target is None:
            continue
        texts.append(TrainedModelFilter.text(row))
        targets.append(target)
        dates.append(row.news_date)
    return texts, targets, dates


def _make_pipeline() -> Pipeline:
    """Build the TF-IDF -> SVD -> scaler -> MLP regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF, sublinear_tf=True, strip_accents="unicode")),
        ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=TRAIN_SEED)),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYERS, alpha=MLP_ALPHA, max_iter=MLP_MAX_ITER, random_state=TRAIN_SEED, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1)),
    ])


def _report_validation(texts: list[str], targets: list[float], dates: list):
    """Fit on the earlier rows, score the held-out later rows, and log ranking metrics."""
    order = np.argsort(np.array(dates))
    texts = [texts[i] for i in order]
    y = np.array(targets, dtype=float)[order]

    split = int(len(texts) * (1.0 - VALIDATION_FRACTION))
    if split < 1 or split >= len(texts):
        logger.warning("Not enough rows for a validation split; skipping validation metrics.")
        return

    train_texts, val_texts = texts[:split], texts[split:]
    train_y, val_y = y[:split], y[split:]

    pipeline = _make_pipeline()
    pipeline.fit(train_texts, train_y)
    predictions = pipeline.predict(val_texts)

    rmse = float(np.sqrt(np.mean((predictions - val_y) ** 2)))
    correlation = spearmanr(predictions, val_y).correlation
    nonzero = val_y != 0
    directional = float(np.mean(np.sign(predictions[nonzero]) == np.sign(val_y[nonzero]))) if nonzero.any() else float("nan")

    logger.info(f"Validation on {len(val_texts)} held-out rows (trained on {len(train_texts)}):")
    logger.info(f"  RMSE: {rmse:.4f} (abnormal 1-day return, %)")
    logger.info(f"  Spearman rank correlation (pred vs actual): {correlation:.4f}")
    logger.info(f"  Directional accuracy: {directional:.4f}")


def main():
    """Train the news-impact network on the datasets the report does not evaluate on."""
    datasets = sorted(DATA_DIR.glob("*.db"))
    if len(datasets) < 2:
        raise SystemExit(f"Need at least 2 datasets in {DATA_DIR} (one to evaluate on, the rest to train on).")

    reserved = max(datasets, key=_row_count)
    training_datasets = [dataset for dataset in datasets if dataset != reserved]
    logger.info(f"Reserving {reserved.name} for evaluation (never trained on) to avoid leakage.")
    logger.info(f"Training on: {', '.join(dataset.name for dataset in training_datasets)}.")

    rows: list[SampledNews] = []
    for dataset in training_datasets:
        rows.extend(_load_rows(dataset))

    texts, targets, dates = _build_examples(rows)
    if len(texts) < 100:
        raise SystemExit(f"Only {len(texts)} labeled rows available; too few to train a model.")
    logger.info(f"Built {len(texts)} labeled training examples.")

    _report_validation(texts, targets, dates)

    logger.info("Refitting the final model on all training examples.")
    pipeline = _make_pipeline()
    pipeline.fit(texts, np.array(targets, dtype=float))

    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, TRAINED_MODEL_PATH)
    logger.info(f"Wrote trained model to {TRAINED_MODEL_PATH}.")

    print(f"\nTrained on {len(texts)} examples from {len(training_datasets)} dataset(s).")
    print(f"Reserved for evaluation: {reserved.name}")
    print(f"Wrote trained model to: {TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
