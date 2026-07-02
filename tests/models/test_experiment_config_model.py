from src.models.experiment_config import ExperimentConfig


def __make() -> ExperimentConfig:
    return ExperimentConfig(
        filter_name="random",
        batch_size=50,
        top_k=10,
        seed=42,
        scalability_sizes=[50, 100, 200],
        subtract_market=True,
        standardize=False,
    )


def test_properties_return_stored_values():
    config = __make()
    assert config.filter_name == "random"
    assert config.batch_size == 50
    assert config.top_k == 10
    assert config.seed == 42
    assert config.scalability_sizes == [50, 100, 200]
    assert config.subtract_market is True
    assert config.standardize is False


def test_to_dict_returns_all_fields():
    assert __make().to_dict() == {
        "filter_name": "random",
        "batch_size": 50,
        "top_k": 10,
        "seed": 42,
        "scalability_sizes": [50, 100, 200],
        "subtract_market": True,
        "standardize": False,
    }
