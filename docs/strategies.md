## Strategies

### BaseStrategy

Module: `quant_strategies.strategies.base_strategy`

Core interface for all strategies:

- `calculate_indicators(data: pd.DataFrame) -> pd.DataFrame` (abstract)
- `generate_signals(data: pd.DataFrame) -> pd.DataFrame` (abstract)
- `backtest(data: pd.DataFrame) -> Dict`
  - Runs indicators, signals, and computes basic counts of BUY/SELL/HOLD
- `validate_data(data: pd.DataFrame) -> bool`
- `get_required_columns() -> List[str]` (default `['open','high','low','close','volume']`)

Signals are represented by `SignalType` enum: `BUY`, `SELL`, `HOLD`.

### BollingerRSIStrategy

Module: `quant_strategies.strategies.bollinger_rsi_strategy`

Mean-reversion using Bollinger Bands and RSI with a volume confirmation filter.

Parameters (defaults in parentheses):

- `bb_period` (20), `bb_std` (2)
- `rsi_period` (14), `rsi_oversold` (30), `rsi_overbought` (70)
- `volume_multiplier` (1.5)

Outputs indicator columns:

- `bb_middle`, `bb_upper`, `bb_lower`, `bb_std`
- `rsi`, `volume_ma`, `volume_ratio`
- `signal`, `signal_strength`

Usage:

```python
from quant_strategies.strategies import BollingerRSIStrategy

strategy = BollingerRSIStrategy({"bb_period": 20, "bb_std": 2})
df_ind = strategy.calculate_indicators(df)
df_sig = strategy.generate_signals(df_ind)
results = strategy.backtest(df)
```

