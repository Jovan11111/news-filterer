# Financial News Ranking Evaluation

Comparative evaluation of approaches for automatically ranking financial news by market impact.

## Running

Every script is configured through JSON files in `config/` — no command-line
arguments. Edit the config, then run the script.

```bash
# generate a dataset (configured in config/dataset_config.json)
python scripts/01_build_dataset.py

# run a filter over the datasets and compare it to the baseline
python scripts/02_filter_news.py

# run a filter across batch sizes and build the PDF report (graphs + tables)
python scripts/03_generate_report.py
```

`config/experiment_config.json` drives scripts `02` and `03`. Key fields under
`experiment`:

- `filter`: which filter the report runs. One of `random`, `rule_based`,
  `vader`, `finbert`, `finbert_cached`, `llm`, `llm_tournament`, `llm_premium`,
  `llm_premium_tournament`.
- `batching.top_k`: how many articles each filter selects.
- `batching.batch_size` / `batching.random_seed`: sampling controls.
- `evaluation.scalability_sizes`: batch sizes the report sweeps.
- `evaluation.subtract_market`: use market-adjusted (abnormal) returns.
- `evaluation.standardize`: divide moves by 30-day volatility (z-scored returns).

The `llm*` filters read their API key from `.env`: `AI_API_KEY` for the standard
variants and `PREMIUM_AI_API_KEY` for the `*_premium` variants.
