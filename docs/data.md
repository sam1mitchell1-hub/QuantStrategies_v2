## Data Collection

### YFinanceClient

Module: `quant_strategies.data.yfinance_client`

- `get_daily_prices(ticker, start_date, end_date) -> pd.DataFrame`
  - Downloads OHLCV via `yfinance.download`
  - Normalizes columns: `date, open, high, low, close, adj_close, volume`
- `save_daily_prices(ticker, start_date, end_date, output_dir="notebooks", filename=None) -> str`
- `get_daily_prices_from_start(ticker, start_date)` and `save_daily_prices_from_start(...)`

Example:

```python
from quant_strategies.data import YFinanceClient
cli = YFinanceClient()
df = cli.get_daily_prices("AAPL", "2020-01-01", "2020-12-31")
```

### DataManager

Module: `quant_strategies.data.data_manager`

Creates directories:

- `data/raw/`, `data/processed/`, `data/metadata/`

Key methods:

- `get_ticker_data(ticker, start_date, output_dir=None) -> str`
- `get_ticker_data_range(ticker, start_date, end_date, output_dir=None) -> str`
- `get_multiple_tickers(tickers, start_date) -> Dict[str, str]`
- `get_multiple_tickers_range(tickers, start_date, end_date) -> Dict[str, str]`
- `get_5year_data(tickers) -> Dict[str, str]`

Each batch run writes a metadata JSON under `data/metadata/` including success/failure lists.

CLI wrapper: `scripts/get_data.py`

```bash
python scripts/get_data.py --tickers AAPL MSFT NVDA --start-date 2019-01-01 --data-dir data
```

### MarketstackClient (optional)

Module: `quant_strategies.data.marketstack_client`

Requires `MARKETSTACK_API_KEY`. Provides `get_daily_prices(ticker, start_date, end_date)` using Marketstack REST API with pagination.

### Configuration: `config/tickers.yaml`

- `demo_tickers`, `tech_tickers`, `sp500_tickers`: example lists
- `defaults`: `start_date`, `data_dir`, `output_format`
- `settings`: `retry_attempts`, `delay_between_requests`, `batch_size`

