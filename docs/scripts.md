## Scripts CLI Guide

### Data Collection

- `scripts/get_data.py`
  - `--tickers TSLA AAPL PLTR`
  - `--start-date YYYY-MM-DD`
  - `--data-dir data`

### Strategy Demo

- `scripts/test_strategy.py`
  - Fetches data (or generates sample) and runs `BollingerRSIStrategy`

### Options

- `scripts/fetch_options.py`
  - `--tickers SPY QQQ`
  - `--num-expiries 3`

- `scripts/explore_options.py`
  - `--ticker SPY`
  - `--kind calls|puts`
  - `--limit 1`
  - `--save-summary`

### IV Plots

- `scripts/plot_iv_curve.py`
  - `--ticker SPY`
  - `--mode smile|term`
  - `--kind calls|puts|both`
  - `--expiry YYYY-MM-DD`

- `scripts/plot_iv_surface.py`
  - `--ticker SPY`
  - `--kind calls|puts`
  - `--num-expiries 8`

### Backtest Entry (placeholder)

- `scripts/run_backtest.py` — scaffold for a richer engine

