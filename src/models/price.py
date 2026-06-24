class Price:
    """Base and forward (1-day, 7-day) closing prices for a ticker on a news date."""

    def __init__(
        self,
        ticker: str,
        base: float | None,
        close_after_1_day: float | None,
        close_after_7_days: float | None,
    ):
        self.__ticker = ticker
        self.__base = base
        self.__close_after_1_day = close_after_1_day
        self.__close_after_7_days = close_after_7_days

    @property
    def ticker(self) -> str:
        return self.__ticker

    @property
    def base(self) -> float | None:
        return self.__base

    @property
    def close_after_1_day(self) -> float | None:
        return self.__close_after_1_day

    @property
    def close_after_7_days(self) -> float | None:
        return self.__close_after_7_days

    @property
    def change_1_day(self) -> float | None:
        return self.__pct(self.__close_after_1_day)

    @property
    def change_7_days(self) -> float | None:
        return self.__pct(self.__close_after_7_days)

    def __pct(self, future: float | None) -> float | None:
        if self.__base is None or future is None or self.__base == 0:
            return None
        return round((future - self.__base) / self.__base * 100.0, 4)
