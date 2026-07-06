import joblib

from src.constants.trained_model import TRAINED_MODEL_PATH
from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainedModelFilter(Filter):
    """Ranks headlines with a neural network trained on this project's own data.

    Unlike the pretrained (FinBERT) and zero-shot (LLM) filters, this one is
    fitted end-to-end on the dataset's labels: a scikit-learn pipeline turns each
    headline into TF-IDF features, reduces them to dense latent-semantic
    components, and feeds a multilayer-perceptron regressor that predicts the
    article's abnormal (market-adjusted) 1-day price move. That single predicted
    move is read per mode exactly like the scalar sentiment filters:

    - BUY prefers the largest predicted rise.
    - SELL prefers the largest predicted fall.
    - AFFECT prefers the largest predicted magnitude either way.

    The trained artifact is produced by scripts/04_train_model.py, which trains
    only on the datasets the report does not evaluate on, so the model never sees
    the rows it is later scored against. The pipeline (~small) is loaded once and
    shared across every instance (one per batch-size run). The filter never looks
    at price data at inference time, so it predicts from text alone.
    """

    # The trained pipeline is loaded once and shared across every instance
    # (one per batch-size run) instead of being reloaded each time.
    __shared_model = None

    def __init__(self, top_k: int = 10, model: object | None = None):
        """Create the filter and obtain the trained model (loaded lazily and shared).

        Args:
            top_k (int): Number of articles to select from each batch.
            model (object | None): Pre-built predictor, mainly for testing; the
                shared on-disk artifact is loaded when None.
        """
        self.__top_k = top_k
        self.__model = model if model is not None else self.__load_model()

    @classmethod
    def __load_model(cls):
        """Load and cache the shared trained pipeline from disk on first use."""
        if cls.__shared_model is None:
            if not TRAINED_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No trained model at {TRAINED_MODEL_PATH}. Run scripts/04_train_model.py first."
                )
            logger.info(f"Loading trained model from {TRAINED_MODEL_PATH}.")
            cls.__shared_model = joblib.load(TRAINED_MODEL_PATH)
        return cls.__shared_model

    @property
    def name(self) -> str:
        """The unique identifier of this filter."""
        return "trained_model"

    @property
    def top_k(self) -> int:
        """The number of articles the filter selects."""
        return self.__top_k

    @staticmethod
    def text(article: SampledNews) -> str:
        """Return the article's title and description joined into one string."""
        return f"{article.news_title} {article.news_description or ''}".strip()

    def __signals(self, articles: list[SampledNews]) -> list[float]:
        """Return the model's predicted move for each article in one batch call."""
        if not articles:
            return []
        texts = [self.text(article) for article in articles]
        return [float(value) for value in self.__model.predict(texts)]

    def signal(self, article: SampledNews) -> float:
        """Return the model's predicted abnormal 1-day price move for one article.

        Args:
            article (SampledNews): The article whose title and description are scored.

        Returns:
            float: The predicted abnormal 1-day price move, in percent.
        """
        return self.__signals([article])[0]

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Rank the batch by the model's prediction and return the top articles for the mode.

        Args:
            articles (list[SampledNews]): The batch of candidate articles.
            mode (FilterMode): Whether to rank for BUY, SELL or AFFECT.

        Returns:
            list[SampledNews]: Up to top_k articles ordered best-first for the mode.
        """
        if len(articles) <= self.__top_k:
            return list(articles)
        signals = self.__signals(list(articles))
        scored = list(zip(articles, signals))
        if mode is FilterMode.BUY:
            scored.sort(key=lambda pair: pair[1], reverse=True)
        elif mode is FilterMode.SELL:
            scored.sort(key=lambda pair: pair[1])
        else:
            scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [article for article, _ in scored[: self.__top_k]]
