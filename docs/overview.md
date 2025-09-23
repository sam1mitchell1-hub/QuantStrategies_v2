## Overview

QuantStrategies is a comprehensive Python platform for quantitative finance research, backtesting, and execution. It includes:

- Core package `quant_strategies` with data clients, strategies, and utilities
- **Stochastic processes framework** for modeling and simulating financial processes
- **PDE solvers framework** for finite difference methods in options pricing
- CLI scripts under `scripts/` for data collection, options fetching, plotting IV curves/surfaces, and strategy testing
- Tests under `tests/` to validate data clients and workflow
- Optional Docker image and a `crontab` for automated options data collection

### Repository Structure

- `quant_strategies/`
  - `data/`: Market data clients and the `DataManager`
  - `strategies/`: Base strategy interface and an example Bollinger+RSI strategy
  - `backtest/`, `execution/`, `utils/`: placeholders for future expansion
- `stochastic/`: Stochastic processes framework
  - `processes/`: Implementations (Geometric Brownian Motion, etc.)
  - `README.md`: Comprehensive documentation
- `pde/`: PDE solvers framework
  - `solvers/`: Finite difference solvers (Black-Scholes Crank-Nicolson, etc.)
  - `utils/`: Grid generation and matrix operations
- `scripts/`: CLI entry points (fetch data, options, plots, strategy demo)
- `tests/`: pytest-based tests
- `config/tickers.yaml`: example tickers and defaults
- `Dockerfile`, `crontab`: containerization and scheduled tasks
- `notebooks/`: research examples

### Key Concepts

- **Data Management**: Ingestion via `yfinance` and optional `marketstack` (requires API key)
- **Trading Strategies**: Interface with `BaseStrategy` and `BollingerRSIStrategy`
- **Stochastic Processes**: Geometric Brownian Motion with fitting and simulation capabilities
- **PDE Solvers**: Finite difference methods for Black-Scholes equation with Crank-Nicolson scheme
- **Options Pricing**: Analytical and numerical methods for European options
- **Risk Management**: Greeks calculation and Monte Carlo analysis
- **Backtesting**: Simple backtest entry point provided by strategies, extensible for PnL/portfolio
- **Visualization**: Options chain tooling and comprehensive plotting scripts

