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
        market_change_1_day: float | None = None,
        market_change_7_days: float | None = None,
        volatility_30_day: float | None = None,
    ):
        """Store the fields of one news row drawn from the dataset.

        Args:
            id (int): Auto-increment primary key of the row in the database.
            news_id (str): Provider-assigned unique id of the article.
            news_title (str): Headline text.
            news_description (str | None): Article summary or body snippet, if any.
            ticker (str): Stock symbol the article refers to.
            news_date (date): Date the article was published.
            price_change_1_day (float | None): 1-day percentage price change.
            price_change_7_days (float | None): 7-day percentage price change.
            market_change_1_day (float | None): 1-day percentage S&P 500 change.
            market_change_7_days (float | None): 7-day percentage S&P 500 change.
            volatility_30_day (float | None): 30-day trailing return volatility, in percent.
        """
        self.__id = id
        self.__news_id = news_id
        self.__news_title = news_title
        self.__news_description = news_description
        self.__ticker = ticker
        self.__news_date = news_date
        self.__price_change_1_day = price_change_1_day
        self.__price_change_7_days = price_change_7_days
        self.__market_change_1_day = market_change_1_day
        self.__market_change_7_days = market_change_7_days
        self.__volatility_30_day = volatility_30_day

    @property
    def id(self) -> int:
        """The auto-increment primary key of the row in the database."""
        return self.__id

    @property
    def news_id(self) -> str:
        """The provider-assigned unique id of the article."""
        return self.__news_id

    @property
    def news_title(self) -> str:
        """The headline text."""
        return self.__news_title

    @property
    def news_description(self) -> str | None:
        """The article summary or body snippet, if any."""
        return self.__news_description

    @property
    def ticker(self) -> str:
        """The stock symbol the article refers to."""
        return self.__ticker

    @property
    def news_date(self) -> date:
        """The date the article was published."""
        return self.__news_date

    @property
    def price_change_1_day(self) -> float | None:
        """The 1-day percentage price change of the stock."""
        return self.__price_change_1_day

    @property
    def price_change_7_days(self) -> float | None:
        """The 7-day percentage price change of the stock."""
        return self.__price_change_7_days

    @property
    def market_change_1_day(self) -> float | None:
        """The 1-day percentage change of the S&P 500 benchmark."""
        return self.__market_change_1_day

    @property
    def market_change_7_days(self) -> float | None:
        """The 7-day percentage change of the S&P 500 benchmark."""
        return self.__market_change_7_days

    @property
    def volatility_30_day(self) -> float | None:
        """The 30-day trailing return volatility of the stock, in percent."""
        return self.__volatility_30_day
