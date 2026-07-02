"""Central home for the project's constants, grouped by domain.

Every non-secret constant lives in one of the themed modules here so there is a
single obvious place to find and change it:

- dataset.py: the news/price data pipeline (Finnhub API, symbols and market
  benchmark, volatility windows and buffers, HTML-cleaning regexes, the SQLite
  schema and column lists, and the dataset output directory).
- filtering.py: the filters (LLM models/keys/prompts, FinBERT model, and the
  rule-based sentiment lexicons).
- evaluation.py: how price moves are scored (trading days spanned per horizon).
- visualization.py: the shared figure palette and layout used by the graphs and
  the PDF report.

Secrets (the API keys) are the only values that stay outside code, in the
environment; their variable names are declared in dataset.py and filtering.py.
"""
