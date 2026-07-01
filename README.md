# Financial News Ranking Evaluation

Comparative evaluation of approaches for automatically ranking financial news by market impact.

## Running

```bash
# generate a dataset
python scripts/01_build_dataset.py

# run a filter over the datasets and compare it to the baseline
python scripts/02_filter_news.py

# run a filter across batch sizes and build the PDF report (graphs + tables)
python scripts/03_generate_report.py

# pick which filter to report on (default: random)
python scripts/03_generate_report.py --filter rule_based
```

Available filters for `--filter`: `random`, `rule_based`, `vader`, `finbert`,
`finbert_cached`, `llm`, `llm_tournament`, `llm_premium`, `llm_premium_tournament`.

The `llm*` filters read their API key from `.env`: `AI_API_KEY` for the standard
variants and `PREMIUM_AI_API_KEY` for the `*_premium` variants.
