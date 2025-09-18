## API Reference

### Package: `quant_strategies`

Re-exports:

- `quant_strategies.data`: `MarketstackClient`, `YFinanceClient`, `DataManager`, `OptionsClient`
- `quant_strategies.strategies`: `BaseStrategy`, `SignalType`, `BollingerRSIStrategy`

### `quant_strategies.data.yfinance_client.YFinanceClient`

- `get_daily_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame`
- `get_daily_prices_from_start(ticker: str, start_date: str) -> pd.DataFrame`
- `save_daily_prices(ticker: str, start_date: str, end_date: str, output_dir: str = "notebooks", filename: str | None = None) -> str`
- `save_daily_prices_from_start(ticker: str, start_date: str, output_dir: str = "notebooks", filename: str | None = None) -> str`

### `quant_strategies.data.data_manager.DataManager`

- `get_ticker_data(...) -> str`
- `get_ticker_data_range(...) -> str`
- `get_multiple_tickers(...) -> Dict[str, str]`
- `get_multiple_tickers_range(...) -> Dict[str, str]`
- `get_5year_data(tickers: List[str]) -> Dict[str, str]`

### `quant_strategies.data.options_client.OptionsClient`

- `list_expirations(ticker: str) -> List[str]`
- `fetch_chain(ticker: str, expiry: str) -> Dict[str, pd.DataFrame>`
- `fetch_and_save(ticker: str, expiries: Optional[List[str]] = None) -> Dict[str, List[str]]`
- `fetch_near_expiries(ticker: str, num_expiries: int = 3) -> Dict[str, List[str]]`

### `quant_strategies.data.marketstack_client.MarketstackClient`

- `get_daily_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame`

### `quant_strategies.strategies.base_strategy.BaseStrategy`

- `calculate_indicators(data: pd.DataFrame) -> pd.DataFrame`
- `generate_signals(data: pd.DataFrame) -> pd.DataFrame`
- `backtest(data: pd.DataFrame) -> Dict`
- `validate_data(data: pd.DataFrame) -> bool`
- `get_required_columns() -> List[str]`

Enums:

- `SignalType` = `BUY | SELL | HOLD`

### `quant_strategies.strategies.bollinger_rsi_strategy.BollingerRSIStrategy`

Constructor:

- `BollingerRSIStrategy(parameters: Optional[Dict] = None)`

Methods:

- Inherits `calculate_indicators`, `generate_signals`, `backtest`
- `get_indicator_columns() -> list`

