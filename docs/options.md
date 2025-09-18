## Options Tools

### OptionsClient

Module: `quant_strategies.data.options_client`

Purpose: fetch option chains via `yfinance.Ticker(...).option_chain(expiry)` with retries, add metadata columns, and save CSVs.

Key methods:

- `list_expirations(ticker) -> List[str]`
- `fetch_chain(ticker, expiry) -> {"calls": DataFrame, "puts": DataFrame}`
- `fetch_and_save(ticker, expiries=None) -> Dict[str, List[str]]`
- `fetch_near_expiries(ticker, num_expiries=3) -> Dict[str, List[str]]`

Files are saved under `data/options/raw/` with pattern: `TICKER_<expiry>_<timestamp>_{calls|puts}.csv`.

### Exploration

Script: `scripts/explore_options.py`

- Lists recent files per ticker/side
- Computes `mid` price and copies `impliedVolatility` to `iv_yf`
- Optional `--save-summary` writes reduced columns under `data/options/processed/`

### Visualization

Scripts:

- `scripts/plot_iv_curve.py` — IV smile and term structure
- `scripts/plot_iv_surface.py` — 3D surface using triangulation

