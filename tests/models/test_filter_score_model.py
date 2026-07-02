from src.models.filter_score import FilterScore
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


def __make() -> FilterScore:
    return FilterScore(
        mode=FilterMode.BUY,
        horizon=PriceHorizon.ONE_DAY,
        top_k=10,
        filter_value=0.6,
        ideal_value=1.0,
        average_value=0.0,
        worst_value=-1.0,
        gap_to_ideal=0.4,
        lift_over_average=0.6,
        normalized_score=0.6,
    )


def test_properties_return_stored_values():
    score = __make()
    assert score.mode is FilterMode.BUY
    assert score.horizon is PriceHorizon.ONE_DAY
    assert score.top_k == 10
    assert score.filter_value == 0.6
    assert score.ideal_value == 1.0
    assert score.average_value == 0.0
    assert score.worst_value == -1.0
    assert score.gap_to_ideal == 0.4
    assert score.lift_over_average == 0.6
    assert score.normalized_score == 0.6


def test_to_dict_serializes_enums_by_value():
    assert __make().to_dict() == {
        "mode": "buy",
        "horizon": "1_day",
        "top_k": 10,
        "filter_value": 0.6,
        "ideal_value": 1.0,
        "average_value": 0.0,
        "worst_value": -1.0,
        "gap_to_ideal": 0.4,
        "lift_over_average": 0.6,
        "normalized_score": 0.6,
    }
