from src.models.price import Price


def __make() -> Price:
    return Price(
        ticker="AAPL",
        base=100.0,
        close_after_1_day=101.0,
        close_after_7_days=110.0,
    )


def test_properties_return_stored_values():
    price = __make()
    assert price.ticker == "AAPL"
    assert price.base == 100.0
    assert price.close_after_1_day == 101.0
    assert price.close_after_7_days == 110.0


def test_changes_are_computed_from_the_base():
    price = __make()
    assert price.change_1_day == 1.0
    assert price.change_7_days == 10.0


def test_changes_are_none_when_a_close_is_missing():
    price = Price(ticker="AAPL", base=None, close_after_1_day=101.0, close_after_7_days=None)
    assert price.change_1_day is None
    assert price.change_7_days is None


def test_to_dict_returns_stored_and_computed_fields():
    assert __make().to_dict() == {
        "ticker": "AAPL",
        "base": 100.0,
        "close_after_1_day": 101.0,
        "close_after_7_days": 110.0,
        "change_1_day": 1.0,
        "change_7_days": 10.0,
    }
