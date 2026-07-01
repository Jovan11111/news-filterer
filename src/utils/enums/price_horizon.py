from enum import Enum


class PriceHorizon(str, Enum):
    """The price-reaction window evaluated for a news article."""

    ONE_DAY = "1_day"
    SEVEN_DAYS = "7_days"
