## Backtesting

The baseline backtest flow is encapsulated by `BaseStrategy.backtest(data)`:

1. `calculate_indicators`
2. `generate_signals`
3. `_calculate_performance` (basic counts)

To extend performance metrics (PnL, drawdown, Sharpe):

- Subclass `BaseStrategy` and override `_calculate_performance`, or
- Build a separate backtest engine in `quant_strategies/backtest/` that consumes `signal` outputs for portfolio simulation.

Data requirements: at minimum `open, high, low, close, volume` and a `date` column for reporting.

Example data prep:

```python
import pandas as pd
df = pd.read_csv("data/raw/AAPL_2019-01-01_to_2024-09-01.csv")
df = df.sort_values("date")
```

Example run:

```python
from quant_strategies.strategies import BollingerRSIStrategy
strategy = BollingerRSIStrategy()
results = strategy.backtest(df)
```

