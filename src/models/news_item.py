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
        """Store the parsed fields of a single news article.

        Args:
            news_id (str): Provider-assigned unique id of the article.
            title (str): Headline text.
            description (str): Article summary or body snippet.
            ticker (str): Stock symbol the article refers to.
            pub_date (date): Date the article was published.
        """
        self.__news_id = news_id
        self.__title = title
        self.__description = description
        self.__ticker = ticker
        self.__pub_date = pub_date

    @property
    def news_id(self) -> str:
        """The provider-assigned unique id of the article."""
        return self.__news_id

    @property
    def title(self) -> str:
        """The headline text."""
        return self.__title

    @property
    def description(self) -> str:
        """The article summary or body snippet."""
        return self.__description

    @property
    def ticker(self) -> str:
        """The stock symbol the article refers to."""
        return self.__ticker

    @property
    def pub_date(self) -> date:
        """The date the article was published."""
        return self.__pub_date
