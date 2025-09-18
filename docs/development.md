## Development & Testing

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[test,notebook]
```

### Running Tests

```bash
pytest -q
```

Key tests:

- `tests/test_yfinance_client.py`: validates column normalization and saving
- `tests/test_data_manager.py`: checks multi-ticker fetch and metadata
- `tests/test_options_client.py`: monkeypatches yfinance to validate CSV outputs

### Code Style

- Prefer descriptive names; avoid cryptic abbreviations
- Add docstrings to public functions/classes

### Extending

- Add new strategies under `quant_strategies/strategies/` by subclassing `BaseStrategy`
- Implement a richer backtesting engine in `quant_strategies/backtest/`
- Contribute scripts under `scripts/` for reproducible workflows

