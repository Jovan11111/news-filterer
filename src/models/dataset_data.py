from datetime import date


class DatasetData:
    """Tickers list name and the date range the dataset is built for."""

    def __init__(self, tickers: str, start_date: date, end_date: date):
        self.__tickers = tickers
        self.__start_date = start_date
        self.__end_date = end_date

    @property
    def tickers(self) -> str:
        return self.__tickers

    @property
    def start_date(self) -> date:
        return self.__start_date

    @property
    def end_date(self) -> date:
        return self.__end_date
