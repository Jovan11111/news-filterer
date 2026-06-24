from datetime import date


class NewsItem:
    """A single parsed news article."""

    def __init__(
        self,
        news_id: str,
        title: str,
        description: str,
        ticker: str,
        pub_date: date,
    ):
        self.__news_id = news_id
        self.__title = title
        self.__description = description
        self.__ticker = ticker
        self.__pub_date = pub_date

    @property
    def news_id(self) -> str:
        return self.__news_id

    @property
    def title(self) -> str:
        return self.__title

    @property
    def description(self) -> str:
        return self.__description

    @property
    def ticker(self) -> str:
        return self.__ticker

    @property
    def pub_date(self) -> date:
        return self.__pub_date
