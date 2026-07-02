class Price:
    """Base and forward (1-day, 7-day) closing prices for a ticker on a news date."""

    def __init__(
        self,
        ticker: str,
        base: float | None,
        close_after_1_day: float | None,
        close_after_7_days: float | None,
    ):
        """Store the base and forward closing prices for a ticker on a news date.

        Args:
            ticker (str): Stock symbol the prices belong to.
            base (float | None): Closing price on the news date, or None if missing.
            close_after_1_day (float | None): Closing price one trading day later.
            close_after_7_days (float | None): Closing price seven calendar days later.
        """
        self.__ticker = ticker
        self.__base = base
        self.__close_after_1_day = close_after_1_day
        self.__close_after_7_days = close_after_7_days

    @property
    def ticker(self) -> str:
        """The stock symbol the prices belong to."""
        return self.__ticker

    @property
    def base(self) -> float | None:
        """The closing price on the news date."""
        return self.__base

    @property
    def close_after_1_day(self) -> float | None:
        """The closing price one trading day after the news date."""
        return self.__close_after_1_day

    @property
    def close_after_7_days(self) -> float | None:
        """The closing price seven calendar days after the news date."""
        return self.__close_after_7_days

    @property
    def change_1_day(self) -> float | None:
        """The 1-day percentage price change relative to the base price."""
        return self.__pct(self.__close_after_1_day)

    @property
    def change_7_days(self) -> float | None:
        """The 7-day percentage price change relative to the base price."""
        return self.__pct(self.__close_after_7_days)

    def to_dict(self) -> dict:
        """Return these prices and their changes as a dict."""
        return {
            "ticker": self.__ticker,
            "base": self.__base,
            "close_after_1_day": self.__close_after_1_day,
            "close_after_7_days": self.__close_after_7_days,
            "change_1_day": self.change_1_day,
            "change_7_days": self.change_7_days,
        }

    def __pct(self, future: float | None) -> float | None:
        """Return the percentage change from the base price to a future price."""
        if self.__base is None or future is None or self.__base == 0:
            return None
        return round((future - self.__base) / self.__base * 100.0, 4)
