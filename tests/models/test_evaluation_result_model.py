from src.models.evaluation_result import EvaluationResult
from src.models.filter_score import FilterScore
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


def __score(mode: FilterMode, horizon: PriceHorizon) -> FilterScore:
    return FilterScore(
        mode=mode,
        horizon=horizon,
        top_k=10,
        filter_value=0.5,
        ideal_value=1.0,
        average_value=0.0,
        worst_value=-1.0,
        gap_to_ideal=0.5,
        lift_over_average=0.5,
        normalized_score=0.5,
    )


def __make() -> EvaluationResult:
    return EvaluationResult(
        filter_name="random",
        batch_size=50,
        top_k=10,
        scores=[
            __score(FilterMode.BUY, PriceHorizon.ONE_DAY),
            __score(FilterMode.SELL, PriceHorizon.SEVEN_DAYS),
        ],
        elapsed_seconds=0.25,
    )


def test_properties_return_stored_values():
    result = __make()
    assert result.filter_name == "random"
    assert result.batch_size == 50
    assert result.top_k == 10
    assert result.elapsed_seconds == 0.25
    assert len(result.scores) == 2


def test_score_looks_up_by_mode_and_horizon():
    result = __make()
    found = result.score(FilterMode.SELL, PriceHorizon.SEVEN_DAYS)
    assert found is not None
    assert found.mode is FilterMode.SELL
    assert found.horizon is PriceHorizon.SEVEN_DAYS
    assert result.score(FilterMode.AFFECT, PriceHorizon.ONE_DAY) is None


def test_to_dict_serializes_nested_scores():
    assert __make().to_dict() == {
        "filter_name": "random",
        "batch_size": 50,
        "top_k": 10,
        "scores": [
            __score(FilterMode.BUY, PriceHorizon.ONE_DAY).to_dict(),
            __score(FilterMode.SELL, PriceHorizon.SEVEN_DAYS).to_dict(),
        ],
        "elapsed_seconds": 0.25,
    }
