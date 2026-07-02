from datetime import date

from src.models.dataset_data import DatasetData


def __make() -> DatasetData:
    return DatasetData(
        tickers="SP100",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )


def test_properties_return_stored_values():
    data = __make()
    assert data.tickers == "SP100"
    assert data.start_date == date(2026, 1, 1)
    assert data.end_date == date(2026, 3, 31)


def test_to_dict_returns_all_fields():
    assert __make().to_dict() == {
        "tickers": "SP100",
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
    }
