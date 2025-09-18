# QuantStrategies_v2

A modular Python project for developing, backtesting, and executing trading strategies.

## Structure

- `quant_strategies/` - Main package with all core modules
- `scripts/` - Entry point scripts for running tasks
- `tests/` - Unit and integration tests
- `notebooks/` - Jupyter notebooks for research and prototyping

## Getting Started

See the full docs in `docs/`:

- Overview: docs/overview.md
- Installation: docs/installation.md
- Quickstart: docs/quickstart.md
- Data: docs/data.md
- Strategies: docs/strategies.md
- Backtesting: docs/backtesting.md
- Options: docs/options.md
- Scripts: docs/scripts.md
- Configuration: docs/configuration.md
- API Reference: docs/api.md
- Development: docs/development.md

Quick install:

```bash
pip install .[notebook]
```

Then try:

```bash
python scripts/get_data.py --tickers TSLA AAPL PLTR --start-date 2019-01-01 --data-dir data
python scripts/test_strategy.py
```