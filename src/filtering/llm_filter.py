import json
import re

from src.constants.filtering import (
    LLM_AZURE_SCOPE,
    LLM_DESCRIPTION_CHARS,
    LLM_MODE_INSTRUCTIONS,
    LLM_PREMIUM_KEY_ENV,
    LLM_PREMIUM_MODEL,
    LLM_STANDARD_KEY_ENV,
    LLM_STANDARD_MODEL,
    LLM_TEMPERATURE,
)
from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.config_provider import ConfigProvider
from src.utils.enums.filter_mode import FilterMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LlmFilter(Filter):
    """Selects news by asking a large language model which headlines matter most.

    The model is prompted with a numbered list of headlines and the active mode,
    and asked to return the item numbers of its top picks. Two axes configure it:

    - is_premium switches between a cheaper standard model (using the
      AI_API_KEY environment variable) and a stronger premium model (using
      PREMIUM_AI_API_KEY).
    - is_tournament switches between scoring the whole batch in one prompt and
      a hierarchical tournament: the batch is split into groups of group_size,
      the model picks the best top_k of each group, and the survivors are
      reduced again until top_k remain. This is the only feasible way to rank
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
        """Create the filter and record its configuration (client stays lazy).

        Args:
            top_k (int): Number of articles to select from each batch.
            is_premium (bool): Use the premium model and key instead of the standard ones.
            is_tournament (bool): Rank very large batches via a hierarchical tournament.
            group_size (int): Group size used per round in tournament mode.
            seed (int): Seed passed to the model for reproducible completions.
            model (str | None): Explicit model name; defaults from is_premium when None.
            client (object | None): Pre-built client, mainly for testing; built lazily otherwise.

        Raises:
            ValueError: If tournament mode is on but group_size is not greater than top_k.
        """
        if is_tournament and group_size <= top_k:
            raise ValueError("group_size must be greater than top_k for tournament selection.")
        self.__top_k = top_k
        self.__is_premium = is_premium
        self.__is_tournament = is_tournament
        self.__group_size = group_size
        self.__seed = seed
        self.__model = model or self.__default_model(is_premium)
        self.__key_env = LLM_PREMIUM_KEY_ENV if is_premium else LLM_STANDARD_KEY_ENV
        self.__client = client
        self.__credential = None
        self.__use_responses = self.__default_use_responses(is_premium)
        self.__prompt_tokens = 0
        self.__completion_tokens = 0
        self.__api_calls = 0

    @staticmethod
    def __default_model(is_premium: bool) -> str:
        """Resolve the model name, letting an env override (e.g. a Foundry deployment) win.

        Reads only non-secret settings from the environment, so it never requires
        an API key; the built-in OpenAI model names are used when nothing is set.
        """
        config = ConfigProvider.get()
        if is_premium:
            return config.premium_ai_model or LLM_PREMIUM_MODEL
        return config.ai_model or LLM_STANDARD_MODEL

    @staticmethod
    def __default_use_responses(is_premium: bool) -> bool:
        """Whether this deployment uses the Responses API instead of Chat Completions."""
        config = ConfigProvider.get()
        return config.premium_ai_use_responses if is_premium else config.ai_use_responses

    @property
    def name(self) -> str:
        """The unique identifier of this filter, reflecting its premium/tournament flags."""
        parts = ["llm"]
        if self.__is_premium:
            parts.append("premium")
        if self.__is_tournament:
            parts.append("tournament")
        return "_".join(parts)

    @property
    def top_k(self) -> int:
        """The number of articles the filter selects."""
        return self.__top_k

    @property
    def is_premium(self) -> bool:
        """Whether the premium model and API key are used."""
        return self.__is_premium

    @property
    def is_tournament(self) -> bool:
        """Whether selection uses the hierarchical tournament."""
        return self.__is_tournament

    @property
    def group_size(self) -> int:
        """The group size used per round in tournament mode."""
        return self.__group_size

    @property
    def model(self) -> str:
        """The name of the model used for completions."""
        return self.__model

    @property
    def prompt_tokens(self) -> int:
        """Total input (prompt) tokens billed across all model calls so far."""
        return self.__prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Total output (completion) tokens billed across all model calls so far."""
        return self.__completion_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) billed across all model calls so far."""
        return self.__prompt_tokens + self.__completion_tokens

    @property
    def api_calls(self) -> int:
        """Number of model calls made across all selections so far."""
        return self.__api_calls

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        """Ask the model which articles matter most and return its top picks for the mode.

        Args:
            articles (list[SampledNews]): The batch of candidate articles.
            mode (FilterMode): Whether to rank for BUY, SELL or AFFECT.

        Returns:
            list[SampledNews]: Up to top_k articles ordered best-first for the mode.
        """
        if len(articles) <= self.__top_k:
            return list(articles)
        if self.__is_tournament:
            return self.__select_tournament(list(articles), mode)
        return self.__select_once(list(articles), mode)

    def __select_tournament(self, articles: list[SampledNews], mode: FilterMode) -> list[SampledNews]:
        """Reduce a large batch to top_k by repeatedly picking the best of each group."""
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
        """Select top_k articles from one group with a single model call."""
        if len(articles) <= self.__top_k:
            return list(articles)
        messages = self.__build_messages(articles, mode)
        content = self.__complete(messages)
        indices = self.__parse_indices(content)
        return self.__gather(articles, indices)

    def __build_messages(self, articles: list[SampledNews], mode: FilterMode) -> list[dict]:
        """Build the system and user chat messages listing the headlines for the mode."""
        system = (
            "You are a financial analyst. You are given a numbered list of news "
            "headlines and must select the ones whose underlying stock is most "
            "likely to react as instructed. Respond ONLY with a JSON array of the "
            "selected item numbers, best first, and nothing else."
        )
        lines = []
        for index, article in enumerate(articles, start=1):
            description = (article.news_description or "")[:LLM_DESCRIPTION_CHARS]
            lines.append(f"{index}. [{article.ticker}] {article.news_title} - {description}".strip())
        body = "\n".join(lines)
        user = (
            f"Select the {self.__top_k} headlines whose stock is most likely to {LLM_MODE_INSTRUCTIONS[mode]}.\n\n"
            f"Headlines:\n{body}\n\n"
            f"Return a JSON array of exactly {self.__top_k} item numbers between 1 and {len(articles)}, best first."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def __complete(self, messages: list[dict]) -> str:
        """Send the chat messages to the model and return its raw text reply."""
        client = self.__get_client()
        if self.__credential is not None:
            # Refresh the Entra token before each call; the credential caches it
            # and only fetches a new one when the current one is near expiry, so
            # long batch runs never fail on an expired token.
            client.api_key = self.__credential.get_token(LLM_AZURE_SCOPE).token
        self.__api_calls += 1
        if self.__use_responses:
            # Responses-API-only deployments (e.g. reasoning/"pro" models) take
            # the same role/content items as `input` and expose the reply text
            # directly; temperature/seed are omitted as such models reject them.
            response = client.responses.create(model=self.__model, input=messages)
            self.__record_usage(response)
            return response.output_text or ""
        response = client.chat.completions.create(model=self.__model, messages=messages, temperature=LLM_TEMPERATURE, seed=self.__seed)
        self.__record_usage(response)
        return response.choices[0].message.content or ""

    def __record_usage(self, response) -> None:
        """Accumulate token usage from a model response, tolerating absent usage."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0)
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0)
        self.__prompt_tokens += int(prompt or 0)
        self.__completion_tokens += int(completion or 0)

    def __get_client(self):
        """Return the API client, building it lazily from the environment on first use.

        When a base URL is configured (e.g. an Azure AI Foundry endpoint) the
        client authenticates with Entra ID via DefaultAzureCredential — no API
        key needed, so an `az login` session is enough. Otherwise it falls back
        to key-based auth against the default OpenAI endpoint.
        """
        if self.__client is None:
            config = ConfigProvider.get()
            base_url = config.premium_ai_base_url if self.__is_premium else config.ai_base_url
            if base_url:
                self.__client = self.__build_azure_client(base_url, config)
            else:
                api_key = config.premium_ai_api_key if self.__is_premium else config.ai_api_key
                if not api_key:
                    raise ValueError(f"Missing required environment variable: {self.__key_env}")
                from openai import OpenAI

                self.__client = OpenAI(api_key=api_key)
        return self.__client

    def __build_azure_client(self, base_url: str, config):
        """Build a keyless OpenAI client for an Azure AI Foundry endpoint (Entra ID)."""
        from azure.identity import DefaultAzureCredential
        from openai import OpenAI

        self.__credential = DefaultAzureCredential()
        options: dict = {
            "api_key": self.__credential.get_token(LLM_AZURE_SCOPE).token,
            "base_url": base_url,
        }
        # Only classic model-inference endpoints need api-version; the newer
        # `/openai/v1/` endpoint is versionless and rejects the query param, so
        # send it only when it is explicitly configured.
        if config.ai_api_version:
            options["default_query"] = {"api-version": config.ai_api_version}
        return OpenAI(**options)

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
        """Map parsed indices to articles, deduping and topping up to top_k."""
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
