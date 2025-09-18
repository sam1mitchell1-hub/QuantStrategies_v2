## Examples

### End-to-end: Historical data to strategy signals

1) Fetch data

```bash
python scripts/get_data.py --tickers AAPL --start-date 2019-01-01 --data-dir data
```

2) Load data and run strategy

```python
import pandas as pd
from quant_strategies.strategies import BollingerRSIStrategy

csv = "data/raw/AAPL_2019-01-01_to_today.csv"
df = pd.read_csv(csv)
df = df.sort_values("date")

strategy = BollingerRSIStrategy()
df_ind = strategy.calculate_indicators(df)
df_sig = strategy.generate_signals(df_ind)
results = strategy.backtest(df)
print(results)
```

### End-to-end: Options chain fetch to IV smile/term structure

1) Fetch options chains

```bash
python scripts/fetch_options.py --tickers SPY --num-expiries 5
```

2) Explore and summarize

```bash
python scripts/explore_options.py --ticker SPY --kind calls --limit 1 --save-summary
```

3) Plot IV

```bash
# Smile for a chosen expiry
python scripts/plot_iv_curve.py --ticker SPY --mode smile --kind both --expiry 2025-09-15

# Term structure across expiries
python scripts/plot_iv_curve.py --ticker SPY --mode term --num-expiries 6

# 3D surface
python scripts/plot_iv_surface.py --ticker SPY --kind calls --num-expiries 8
```

### Docker + cron: Scheduled options fetch

Build the image and run with mounted `data/` and `logs/` directories:

```bash
docker build -t quant-strategies:latest .
mkdir -p data logs
docker run --rm -it \
  -v "$PWD/data":/app/data \
  -v "$PWD/logs":/app/logs \
  quant-strategies:latest
```

The default command runs `supercronic` using the provided `crontab` to fetch SPY/QQQ options daily at 07:05 UTC.

