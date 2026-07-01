import json
import os
import re

from dotenv import load_dotenv

from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.logger import get_logger

logger = get_logger(__name__)

_STANDARD_MODEL = "gpt-4o-mini"
_PREMIUM_MODEL = "gpt-4o"
_STANDARD_KEY_ENV = "AI_API_KEY"
_PREMIUM_KEY_ENV = "PREMIUM_AI_API_KEY"

_DESCRIPTION_CHARS = 200

_INSTRUCTIONS = {
    FilterMode.BUY: "rise the most (strongest positive price reaction)",
    FilterMode.SELL: "fall the most (strongest negative price reaction)",
    FilterMode.AFFECT: "move the most in either direction (largest absolute price reaction)",
}


class LlmFilter(Filter):
    """Selects news by asking a large language model which headlines matter most.

    The model is prompted with a numbered list of headlines and the active mode,
    and asked to return the item numbers of its top picks. Two axes configure it:

    - ``is_premium`` switches between a cheaper standard model (using the
      ``AI_API_KEY`` environment variable) and a stronger premium model (using
      ``PREMIUM_AI_API_KEY``).
    - ``is_tournament`` switches between scoring the whole batch in one prompt and
      a hierarchical tournament: the batch is split into groups of ``group_size``,
      the model picks the best ``top_k`` of each group, and the survivors are
      reduced again until ``top_k`` remain. This is the only feasible way to rank
      very large batches that cannot fit in a single prompt, and—unlike the
      scalar sentiment filters—it can change the result because the model ranks
      headlines relative to the others in the same prompt.

    The client is created lazily on first use, so constructing the filter never
    requires an API key; only an actual selection does. The filter never looks at
    price data, so it predicts from text alone.
    """

    def __init__(
        self,
        top_k: int = 10,
        is_premium: bool = False,
        is_tournament: bool = False,
        group_size: int = 50,
        seed: int = 42,
        model: str | None = None,
        client: object | None = None,
    ):
        if is_tournament and group_size <= top_k:
            raise ValueError("group_size must be greater than top_k for tournament selection.")
        self.__top_k = top_k
        self.__is_premium = is_premium
        self.__is_tournament = is_tournament
        self.__group_size = group_size
        self.__seed = seed
        self.__model = model or (_PREMIUM_MODEL if is_premium else _STANDARD_MODEL)
        self.__key_env = _PREMIUM_KEY_ENV if is_premium else _STANDARD_KEY_ENV
        self.__client = client

    @property
    def name(self) -> str:
        parts = ["llm"]
        if self.__is_premium:
            parts.append("premium")
        if self.__is_tournament:
            parts.append("tournament")
        return "_".join(parts)

    @property
    def top_k(self) -> int:
        return self.__top_k

    @property
    def is_premium(self) -> bool:
        return self.__is_premium

    @property
    def is_tournament(self) -> bool:
        return self.__is_tournament

    @property
    def group_size(self) -> int:
        return self.__group_size

    @property
    def model(self) -> str:
        return self.__model

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        if len(articles) <= self.__top_k:
            return list(articles)
        if self.__is_tournament:
            return self.__select_tournament(list(articles), mode)
        return self.__select_once(list(articles), mode)

    def __select_tournament(self, articles: list[SampledNews], mode: FilterMode) -> list[SampledNews]:
        pool = articles
        while len(pool) > self.__group_size:
            survivors: list[SampledNews] = []
            for start in range(0, len(pool), self.__group_size):
                group = pool[start : start + self.__group_size]
                survivors.extend(self.__select_once(group, mode))
            pool = survivors
        if len(pool) > self.__top_k:
            return self.__select_once(pool, mode)
        return pool

    def __select_once(self, articles: list[SampledNews], mode: FilterMode) -> list[SampledNews]:
        if len(articles) <= self.__top_k:
            return list(articles)
        messages = self.__build_messages(articles, mode)
        content = self.__complete(messages)
        indices = self.__parse_indices(content)
        return self.__gather(articles, indices)

    def __build_messages(self, articles: list[SampledNews], mode: FilterMode) -> list[dict]:
        system = (
            "You are a financial analyst. You are given a numbered list of news "
            "headlines and must select the ones whose underlying stock is most "
            "likely to react as instructed. Respond ONLY with a JSON array of the "
            "selected item numbers, best first, and nothing else."
        )
        lines = []
        for index, article in enumerate(articles, start=1):
            description = (article.news_description or "")[:_DESCRIPTION_CHARS]
            lines.append(f"{index}. [{article.ticker}] {article.news_title} - {description}".strip())
        body = "\n".join(lines)
        user = (
            f"Select the {self.__top_k} headlines whose stock is most likely to {_INSTRUCTIONS[mode]}.\n\n"
            f"Headlines:\n{body}\n\n"
            f"Return a JSON array of exactly {self.__top_k} item numbers between 1 and {len(articles)}, best first."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def __complete(self, messages: list[dict]) -> str:
        client = self.__get_client()
        response = client.chat.completions.create(model=self.__model, messages=messages, temperature=0, seed=self.__seed)
        return response.choices[0].message.content or ""

    def __get_client(self):
        if self.__client is None:
            load_dotenv()
            api_key = os.getenv(self.__key_env)
            if not api_key:
                raise ValueError(f"Missing required environment variable: {self.__key_env}")
            from openai import OpenAI

            self.__client = OpenAI(api_key=api_key)
        return self.__client

    @staticmethod
    def __parse_indices(content: str) -> list[int]:
        """Parse 1-based item numbers from the model's reply into 0-based indices."""
        text = content.strip()
        data = None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            for key in ("selected", "indices", "ids", "items", "articles"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        numbers: list[int] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, bool):
                    continue
                if isinstance(item, int):
                    numbers.append(item)
                elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
                    numbers.append(int(item))
        if not numbers:
            numbers = [int(match) for match in re.findall(r"\d+", text)]
        return [number - 1 for number in numbers]

    def __gather(self, articles: list[SampledNews], indices: list[int]) -> list[SampledNews]:
        """Map parsed indices to articles, deduping and topping up to ``top_k``."""
        chosen: list[SampledNews] = []
        seen: set[int] = set()
        for index in indices:
            if 0 <= index < len(articles) and index not in seen:
                chosen.append(articles[index])
                seen.add(index)
                if len(chosen) == self.__top_k:
                    return chosen
        for index, article in enumerate(articles):
            if index not in seen:
                chosen.append(article)
                seen.add(index)
                if len(chosen) == self.__top_k:
                    break
        return chosen
