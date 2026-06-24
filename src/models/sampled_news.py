from datetime import date


class SampledNews:
    """One news row drawn from the dataset."""

    def __init__(
        self,
        id: int,
        news_id: str,
        news_title: str,
        news_description: str | None,
        ticker: str,
        news_date: date,
        price_change_1_day: float | None,
        price_change_7_days: float | None,
    ):
        self.__id = id
        self.__news_id = news_id
        self.__news_title = news_title
        self.__news_description = news_description
        self.__ticker = ticker
        self.__news_date = news_date
        self.__price_change_1_day = price_change_1_day
        self.__price_change_7_days = price_change_7_days

    @property
    def id(self) -> int:
        return self.__id

    @property
    def news_id(self) -> str:
        return self.__news_id

    @property
    def news_title(self) -> str:
        return self.__news_title

    @property
    def news_description(self) -> str | None:
        return self.__news_description

    @property
    def ticker(self) -> str:
        return self.__ticker

    @property
    def news_date(self) -> date:
        return self.__news_date

    @property
    def price_change_1_day(self) -> float | None:
        return self.__price_change_1_day

    @property
    def price_change_7_days(self) -> float | None:
        return self.__price_change_7_days
