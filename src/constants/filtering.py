"""Constants for the news filters (LLM, FinBERT and the rule-based lexicon)."""

import re

from src.utils.enums.filter_mode import FilterMode

# --- LLM filter ---------------------------------------------------------------

# Cheaper standard model and the environment variable holding its (secret) key.
LLM_STANDARD_MODEL = "gpt-4o-mini"
LLM_STANDARD_KEY_ENV = "AI_API_KEY"

# Stronger premium model and the environment variable holding its (secret) key.
LLM_PREMIUM_MODEL = "gpt-4o"
LLM_PREMIUM_KEY_ENV = "PREMIUM_AI_API_KEY"

# Optional OpenAI-compatible base URLs. When set they redirect the standard or
# premium client to a custom endpoint (e.g. an Azure AI Foundry "models"
# endpoint), while an empty/absent value keeps the default OpenAI endpoint.
LLM_STANDARD_BASE_URL_ENV = "AI_BASE_URL"
LLM_PREMIUM_BASE_URL_ENV = "PREMIUM_AI_BASE_URL"

# Optional model/deployment name overrides. On Foundry the "model" is the
# deployment name you chose (e.g. a Claude deployment), which need not match the
# default OpenAI model names above.
LLM_STANDARD_MODEL_ENV = "AI_MODEL"
LLM_PREMIUM_MODEL_ENV = "PREMIUM_AI_MODEL"

# Optional API version sent as the `api-version` query parameter, required by
# classic Azure model-inference endpoints and left off for the versionless
# `/openai/v1/` endpoint, which rejects it.
LLM_API_VERSION_ENV = "AI_API_VERSION"

# Optional flags selecting the Responses API instead of Chat Completions. Some
# newer deployments (e.g. reasoning/"pro" models) only speak the Responses API.
LLM_STANDARD_USE_RESPONSES_ENV = "AI_USE_RESPONSES"
LLM_PREMIUM_USE_RESPONSES_ENV = "PREMIUM_AI_USE_RESPONSES"

# Entra ID token scope used for keyless auth against Azure AI Foundry / Azure
# OpenAI endpoints (via DefaultAzureCredential, e.g. an `az login` session).
LLM_AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"

# Number of description characters included per headline in the prompt.
LLM_DESCRIPTION_CHARS = 200

# Sampling temperature; 0 keeps completions as deterministic as possible.
LLM_TEMPERATURE = 0

# Per-mode phrasing appended to the prompt describing the reaction to rank for.
LLM_MODE_INSTRUCTIONS = {
    FilterMode.BUY: "rise the most (strongest positive price reaction)",
    FilterMode.SELL: "fall the most (strongest negative price reaction)",
    FilterMode.AFFECT: "move the most in either direction (largest absolute price reaction)",
}

# --- FinBERT filter -----------------------------------------------------------

# Hugging Face model id for the financial-sentiment classifier.
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Number of headlines FinBERT scores per inference batch.
FINBERT_INFERENCE_BATCH_SIZE = 32

# --- Rule-based filter --------------------------------------------------------

# Tokenizer that splits lower-cased text into alphabetic words.
RULE_TOKEN_RE = re.compile(r"[a-z]+")

# Words that, in financial headlines, lean bullish. Stored as a flat set of
# lower-case tokens (including common inflections) so a single membership test
# is enough after the text is tokenised.
POSITIVE_TERMS = frozenset({
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
NEGATIVE_TERMS = frozenset({
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
