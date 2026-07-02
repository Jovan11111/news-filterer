from datetime import date


class DatasetData:
    """Tickers list name and the date range the dataset is built for."""

    def __init__(self, tickers: str, start_date: date, end_date: date):
        """Store the tickers list name and date range a dataset is built for.

        Args:
            tickers (str): Name of the ticker list to build the dataset from.
            start_date (date): First date (inclusive) of the news range.
            end_date (date): Last date (inclusive) of the news range.
        """
        self.__tickers = tickers
        self.__start_date = start_date
        self.__end_date = end_date

    @property
    def tickers(self) -> str:
        """The name of the ticker list to build the dataset from."""
        return self.__tickers

    @property
    def start_date(self) -> date:
        """The first date (inclusive) of the news range."""
        return self.__start_date

    @property
    def end_date(self) -> date:
        """The last date (inclusive) of the news range."""
        return self.__end_date

    def to_dict(self) -> dict:
        """Return this dataset spec as a dict."""
        return {
            "tickers": self.__tickers,
            "start_date": self.__start_date.isoformat(),
            "end_date": self.__end_date.isoformat(),
        }
