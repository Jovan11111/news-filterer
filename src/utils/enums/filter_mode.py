from enum import Enum


class FilterMode(str, Enum):
    """The instruction given to a filter for a single selection run."""

    BUY = "buy"
    SELL = "sell"
    AFFECT = "affect"
