## Quickstart

### 1) Install

```bash
pip install .[notebook]
```

### 2) Fetch historical OHLCV data

```bash
python scripts/get_data.py --tickers TSLA AAPL PLTR --start-date 2019-01-01 --data-dir data
```

Outputs:

- CSVs under `data/raw/`
- Metadata JSON under `data/metadata/`

### 3) Run the example strategy demo

```bash
python scripts/test_strategy.py
```

Shows indicator calculation, signal generation, and simple backtest counts.

### 4) Fetch options chains and explore

```bash
python scripts/fetch_options.py --tickers SPY QQQ --num-expiries 3
python scripts/explore_options.py --ticker SPY --kind calls --limit 1 --save-summary
```

### 5) Plot IV curves and surfaces

```bash
python scripts/plot_iv_curve.py --ticker SPY --mode smile --kind both --expiry 2025-09-15
python scripts/plot_iv_curve.py --ticker QQQ --mode term --num-expiries 6
python scripts/plot_iv_surface.py --ticker SPY --kind calls --num-expiries 8
```

