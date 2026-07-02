"""Constants for scoring news price moves during evaluation."""

from src.utils.enums.price_horizon import PriceHorizon

# Approximate number of trading days spanned by each horizon, used to scale the
# daily volatility up to the horizon when standardizing a price move into a
# z-score (a 7-day window covers roughly five trading sessions).
HORIZON_TRADING_DAYS = {PriceHorizon.ONE_DAY: 1, PriceHorizon.SEVEN_DAYS: 5}
