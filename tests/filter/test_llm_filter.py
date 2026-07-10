from datetime import date
from types import SimpleNamespace

import pytest

from src.filtering.llm_filter import LlmFilter
from src.models.sampled_news import SampledNews
from src.utils.app_config import AppConfig
from src.utils.config_provider import ConfigProvider
from src.utils.enums.filter_mode import FilterMode


def make_article(article_id: int, title: str, description: str | None = "") -> SampledNews:
    """Build a SampledNews with only the text fields that the filter reads."""
    return SampledNews(
        id=article_id,
        news_id=f"n{article_id}",
        news_title=title,
        news_description=description,
        ticker="TEST",
        news_date=date(2025, 1, 1),
        price_change_1_day=None,
        price_change_7_days=None,
    )


class StubClient:
    """A fake OpenAI-style client that returns a canned reply without any network.

    content may be a string or a callable receiving the create() kwargs, so a
    test can inspect the prompt (e.g. the number of headlines) and respond.
    """

    def __init__(self, content):
        self.__content = content
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.__content(kwargs) if callable(self.__content) else self.__content
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _batch(count: int) -> list[SampledNews]:
    return [make_article(i, f"Headline number {i}") for i in range(1, count + 1)]


def test_name_combinations():
    assert LlmFilter().name == "llm"
    assert LlmFilter(is_tournament=True).name == "llm_tournament"
    assert LlmFilter(is_premium=True).name == "llm_premium"
    assert LlmFilter(is_premium=True, is_tournament=True).name == "llm_premium_tournament"


def test_premium_selects_stronger_model(monkeypatch):
    # Isolate from any real .env so the built-in default models are exercised.
    monkeypatch.setattr(
        ConfigProvider,
        "_config",
        AppConfig(finnhub_api_key=None, ai_api_key=None, premium_ai_api_key=None),
    )
    assert LlmFilter().model == "gpt-4o-mini"
    assert LlmFilter(is_premium=True).model == "gpt-4o"


def test_model_override_from_config(monkeypatch):
    # An env override (e.g. a Foundry deployment name) wins over the defaults.
    monkeypatch.setattr(
        ConfigProvider,
        "_config",
        AppConfig(
            finnhub_api_key=None,
            ai_api_key=None,
            premium_ai_api_key=None,
            ai_model="std-deployment",
            premium_ai_model="foundry-deployment",
        ),
    )
    assert LlmFilter().model == "std-deployment"
    assert LlmFilter(is_premium=True).model == "foundry-deployment"


def test_tournament_requires_group_larger_than_top_k():
    with pytest.raises(ValueError):
        LlmFilter(top_k=10, is_tournament=True, group_size=10)


def test_select_returns_all_without_calling_model_when_small():
    stub = StubClient("[1,2,3]")
    batch = _batch(5)
    selected = LlmFilter(top_k=10, client=stub).select(batch, FilterMode.BUY)
    assert selected == batch
    assert stub.calls == []


def test_select_uses_model_chosen_indices():
    stub = StubClient("[3, 1, 2]")
    batch = _batch(20)
    selected = LlmFilter(top_k=3, client=stub).select(batch, FilterMode.BUY)
    assert [a.id for a in selected] == [3, 1, 2]


def test_select_parses_json_object_reply():
    stub = StubClient('{"selected": [5, 4]}')
    batch = _batch(20)
    selected = LlmFilter(top_k=2, client=stub).select(batch, FilterMode.SELL)
    assert [a.id for a in selected] == [5, 4]


def test_select_tops_up_when_model_returns_too_few():
    stub = StubClient("[2]")
    batch = _batch(20)
    selected = LlmFilter(top_k=3, client=stub).select(batch, FilterMode.BUY)
    # First the model's pick (2), then the batch order fills the rest (1, 3).
    assert [a.id for a in selected] == [2, 1, 3]


def test_select_ignores_invalid_and_duplicate_indices():
    stub = StubClient("[999, 2, 2, 0, 3]")
    batch = _batch(20)
    selected = LlmFilter(top_k=3, client=stub).select(batch, FilterMode.BUY)
    assert [a.id for a in selected] == [2, 3, 1]


def test_select_falls_back_to_order_on_garbage_reply():
    stub = StubClient("I cannot help with that.")
    batch = _batch(20)
    selected = LlmFilter(top_k=3, client=stub).select(batch, FilterMode.BUY)
    assert [a.id for a in selected] == [1, 2, 3]


def test_select_does_not_mutate_input():
    stub = StubClient("[3, 1, 2]")
    batch = _batch(20)
    original = list(batch)
    LlmFilter(top_k=3, client=stub).select(batch, FilterMode.BUY)
    assert batch == original


def test_tournament_reduces_in_rounds():
    # The stub always picks the first top_k of whatever group it is shown.
    stub = StubClient(lambda kwargs: "[1,2,3,4,5,6,7,8,9,10]")
    batch = _batch(120)
    selected = LlmFilter(top_k=10, is_tournament=True, group_size=50, client=stub).select(batch, FilterMode.BUY)
    assert len(selected) == 10
    # 120 -> groups [50,50,20] = 3 calls, survivors 30 -> final call = 4 calls total.
    assert len(stub.calls) == 4


def test_tournament_matches_regular_when_model_ranks_consistently():
    stub_regular = StubClient(lambda kwargs: "[1,2,3,4,5,6,7,8,9,10]")
    stub_tournament = StubClient(lambda kwargs: "[1,2,3,4,5,6,7,8,9,10]")
    batch = _batch(60)
    regular = LlmFilter(top_k=10, client=stub_regular).select(batch, FilterMode.BUY)
    tournament = LlmFilter(top_k=10, is_tournament=True, group_size=50, client=stub_tournament).select(batch, FilterMode.BUY)
    assert [a.id for a in regular] == [a.id for a in tournament] == list(range(1, 11))


def test_prompt_contains_mode_instruction_and_headlines():
    captured = {}

    def content(kwargs):
        captured["messages"] = kwargs["messages"]
        return "[1,2,3]"

    stub = StubClient(content)
    batch = _batch(20)
    LlmFilter(top_k=3, client=stub).select(batch, FilterMode.SELL)
    user_message = captured["messages"][1]["content"]
    assert "fall the most" in user_message
    assert "Headline number 1" in user_message
