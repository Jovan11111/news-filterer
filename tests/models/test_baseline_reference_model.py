from datetime import date

from src.models.baseline_reference import BaselineReference
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


def __article() -> SampledNews:
    return SampledNews(
        id=1,
        news_id="n1",
        news_title="Title",
        news_description="Description",
        ticker="AAPL",
        news_date=date(2026, 1, 2),
        price_change_1_day=1.5,
        price_change_7_days=2.5,
    )


def __make() -> BaselineReference:
    return BaselineReference(
        mode=FilterMode.SELL,
        horizon=PriceHorizon.SEVEN_DAYS,
        top_k=10,
        ideal_value=1.0,
        average_value=0.0,
        worst_value=-1.0,
        ideal_selection=[__article()],
    )


def test_properties_return_stored_values():
    reference = __make()
    assert reference.mode is FilterMode.SELL
    assert reference.horizon is PriceHorizon.SEVEN_DAYS
    assert reference.top_k == 10
    assert reference.ideal_value == 1.0
    assert reference.average_value == 0.0
    assert reference.worst_value == -1.0
    assert len(reference.ideal_selection) == 1
    assert reference.ideal_selection[0].news_id == "n1"


def test_to_dict_serializes_enums_and_nested_selection():
    assert __make().to_dict() == {
        "mode": "sell",
        "horizon": "7_days",
        "top_k": 10,
        "ideal_value": 1.0,
        "average_value": 0.0,
        "worst_value": -1.0,
        "ideal_selection": [__article().to_dict()],
    }
