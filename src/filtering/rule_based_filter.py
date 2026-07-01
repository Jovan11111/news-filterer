import re

from src.filtering.filter import Filter
from src.models.sampled_news import SampledNews
from src.utils.enums.filter_mode import FilterMode

_TOKEN_RE = re.compile(r"[a-z]+")

# Words that, in financial headlines, lean bullish. Stored as a flat set of
# lower-case tokens (including common inflections) so a single membership test
# is enough after the text is tokenised.
_POSITIVE_TERMS = frozenset({
    "beat", "beats", "beating", "surge", "surges", "surged", "soar", "soars", "soared",
    "rally", "rallies", "rallied", "gain", "gains", "gained", "jump", "jumps", "jumped",
    "rise", "rises", "rose", "rising", "climb", "climbs", "climbed", "upgrade", "upgrades",
    "upgraded", "outperform", "outperforms", "outperformed", "record", "profit", "profits",
    "profitable", "growth", "strong", "strength", "boost", "boosts", "boosted", "raise",
    "raises", "raised", "bullish", "buyback", "buybacks", "dividend", "dividends", "expansion",
    "win", "wins", "won", "approval", "approved", "approves", "breakthrough", "rebound",
    "rebounds", "rebounded", "optimistic", "upbeat", "top", "tops", "topped", "exceed",
    "exceeds", "exceeded", "higher", "soaring", "surging", "momentum", "accelerate",
    "accelerates", "accelerated", "expand", "expands", "expanded", "outpace", "outpaces",
})

# Words that lean bearish, mirroring the positive set.
_NEGATIVE_TERMS = frozenset({
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "slump", "slumps", "slumped",
    "drop", "drops", "dropped", "fall", "falls", "fell", "falling", "decline", "declines",
    "declined", "downgrade", "downgrades", "downgraded", "loss", "losses", "weak", "weakness",
    "weaker", "cut", "cuts", "lawsuit", "sue", "sues", "sued", "fraud", "probe", "investigation",
    "recall", "recalls", "layoff", "layoffs", "bearish", "warn", "warns", "warned", "warning",
    "slash", "slashes", "slashed", "tumble", "tumbles", "tumbled", "sink", "sinks", "sank",
    "crash", "crashes", "crashed", "bankruptcy", "default", "lower", "selloff", "underperform",
    "underperforms", "downturn", "halt", "halts", "halted", "delay", "delays", "delayed",
    "concern", "concerns", "risk", "risks", "fear", "fears", "shortfall", "disappoint",
    "disappoints", "disappointing", "scandal", "sinking", "tumbling", "plunging",
})


class RuleBasedFilter(Filter):
    """A keyword-driven filter that scores each headline's directional sentiment.

    Every article is reduced to an integer signal: the number of bullish keywords
    minus the number of bearish keywords found in its title and description. The
    three modes then read that single signal differently:

    - ``BUY`` prefers the most positive signals (expected to rise).
    - ``SELL`` prefers the most negative signals (expected to fall).
    - ``AFFECT`` prefers the largest magnitude either way (expected to move most).

    The filter never looks at price data, so it makes a genuine prediction from
    text alone rather than peeking at the labels it is evaluated against.
    """

    def __init__(self, top_k: int = 10):
        self.__top_k = top_k

    @property
    def name(self) -> str:
        return "rule_based"

    @property
    def top_k(self) -> int:
        return self.__top_k

    def signal(self, article: SampledNews) -> int:
        """Return the bullish-minus-bearish keyword score for one article."""
        text = f"{article.news_title} {article.news_description or ''}".lower()
        tokens = _TOKEN_RE.findall(text)
        positive = sum(1 for token in tokens if token in _POSITIVE_TERMS)
        negative = sum(1 for token in tokens if token in _NEGATIVE_TERMS)
        return positive - negative

    def select(
        self, articles: list[SampledNews], mode: FilterMode
    ) -> list[SampledNews]:
        if len(articles) <= self.__top_k:
            return list(articles)
        scored = [(article, self.signal(article)) for article in articles]
        if mode is FilterMode.BUY:
            scored.sort(key=lambda pair: pair[1], reverse=True)
        elif mode is FilterMode.SELL:
            scored.sort(key=lambda pair: pair[1])
        else:
            scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [article for article, _ in scored[: self.__top_k]]
