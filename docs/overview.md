## Overview

QuantStrategies is a modular Python project for researching, backtesting, and executing trading strategies. It includes:

- Core package `quant_strategies` with data clients, strategies, and utilities
- CLI scripts under `scripts/` for data collection, options fetching, plotting IV curves/surfaces, and strategy testing
- Tests under `tests/` to validate data clients and workflow
- Optional Docker image and a `crontab` for automated options data collection

### Repository Structure

- `quant_strategies/`
  - `data/`: Market data clients and the `DataManager`
  - `strategies/`: Base strategy interface and an example Bollinger+RSI strategy
  - `backtest/`, `execution/`, `utils/`: placeholders for future expansion
- `scripts/`: CLI entry points (fetch data, options, plots, strategy demo)
- `tests/`: pytest-based tests
- `config/tickers.yaml`: example tickers and defaults
- `Dockerfile`, `crontab`: containerization and scheduled tasks
- `notebooks/`: research examples

### Key Concepts

- Data ingestion via `yfinance` and optional `marketstack` (requires API key)
- Strategy interface with `BaseStrategy` and `BollingerRSIStrategy`
- Simple backtest entry point provided by strategies, extensible for PnL/portfolio
- Options chain tooling and visualization scripts

