from src.models.evaluation_result import EvaluationResult
from src.models.filter_run import FilterRun
from src.models.filter_score import FilterScore
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon


def __evaluation() -> EvaluationResult:
    score = FilterScore(
        mode=FilterMode.BUY,
        horizon=PriceHorizon.ONE_DAY,
        top_k=10,
        filter_value=0.5,
        ideal_value=1.0,
        average_value=0.0,
        worst_value=-1.0,
        gap_to_ideal=0.5,
        lift_over_average=0.5,
        normalized_score=0.5,
    )
    return EvaluationResult(
        filter_name="random",
        batch_size=50,
        top_k=10,
        scores=[score],
        elapsed_seconds=0.25,
    )


def __make() -> FilterRun:
    return FilterRun(requested_size=100, evaluation=__evaluation())


def test_properties_return_stored_values():
    run = __make()
    assert run.requested_size == 100
    assert run.evaluation.filter_name == "random"
    assert run.evaluation.batch_size == 50


def test_to_dict_serializes_nested_evaluation():
    assert __make().to_dict() == {
        "requested_size": 100,
        "evaluation": __evaluation().to_dict(),
    }
