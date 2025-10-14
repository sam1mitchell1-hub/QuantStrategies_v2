# Live Trading Infrastructure Review Checklist

**Branch:** `feature/oms-portfolio-system`  
**Reviewer:** _____________  
**Review Date:** _____________

## 📋 Overview

This checklist reviews the live trading infrastructure built on top of the OMS framework. Verify each component works correctly with real stock data from yfinance.

---

## ✅ Component Implementation Status

### 1. Live Data Fetcher (`quant_strategies/data/live_data_fetcher.py` - 285 lines)

**Core Requirements:**
- [ ] `get_liquid_stocks()` returns list of 20 liquid stock tickers
- [ ] `get_stock_name(ticker)` returns company names
- [ ] `get_current_prices(tickers)` fetches prices for multiple tickers
- [ ] `get_market_data(ticker)` returns OMS-compatible format
- [ ] `get_historical_data(ticker, period, interval)` fetches OHLCV data
- [ ] `get_latest_bars(ticker, n)` returns last N bars for indicators

**Liquid Stocks List:**
- [ ] Contains AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA (Tech)
- [ ] Contains JPM, BAC, GS, V, MA (Finance)
- [ ] Contains WMT, PG, KO, PEP (Consumer)
- [ ] Contains JNJ, UNH, PFE (Healthcare)
- [ ] Total of ~20 highly liquid stocks

**Caching System:**
- [ ] Price cache works (60s default)
- [ ] Historical cache works (5min for hist data)
- [ ] `clear_cache()` clears all cached data
- [ ] `get_cache_stats()` returns cache statistics
- [ ] Cache prevents excessive API calls

**Market Data Format:**
- [ ] Returns dict with: ticker, last, bid, ask, close, volume, timestamp
- [ ] Bid/ask calculated if not available (0.1% spread)
- [ ] Compatible with OMS.process_signal() format
- [ ] Handles missing data gracefully

**Error Handling:**
- [ ] Handles ticker not found errors
- [ ] Handles yfinance API failures
- [ ] Logs warnings for missing data
- [ ] Returns None instead of crashing

**Test These:**
```python
from quant_strategies.data.live_data_fetcher import LiveDataFetcher

fetcher = LiveDataFetcher()

# Test 1: Get liquid stocks
stocks = fetcher.get_liquid_stocks()
print(f"Liquid stocks: {len(stocks)}")  # Should be ~20

# Test 2: Get market data
data = fetcher.get_market_data('AAPL')
print(f"AAPL: ${data['last']:.2f}")  # Should show current price

# Test 3: Get historical data
hist = fetcher.get_historical_data('AAPL', period='1mo')
print(f"Bars: {len(hist)}")  # Should show ~20-30 daily bars

# Test 4: Cache stats
stats = fetcher.get_cache_stats()
print(stats)  # Should show cache sizes
```

**Review Notes:**
```
[Add your comments here]




```

**Status:** [ ] Approved [ ] Needs Changes [ ] Questions

---

### 2. Live Strategy Runner (`quant_strategies/strategies/live_strategy_runner.py` - 289 lines)

**Core Requirements:**
- [ ] `__init__()` accepts strategy, oms, data_fetcher, watchlist
- [ ] `run_cycle()` processes all tickers in watchlist
- [ ] Returns cycle results dict with signals/orders/errors
- [ ] `_process_ticker()` handles single ticker workflow
- [ ] `_execute_signal()` converts to OMS signal and executes

**Signal Processing:**
- [ ] Fetches historical data for indicators
- [ ] Validates data has required columns
- [ ] Calculates indicators via strategy
- [ ] Generates signals via strategy
- [ ] Extracts latest signal from DataFrame

**Duplicate Prevention:**
- [ ] Tracks last signal per ticker
- [ ] Skips if signal unchanged (avoids duplicate orders)
- [ ] Resets when signal changes
- [ ] `reset_signal_memory()` clears tracking

**OMS Integration:**
- [ ] Calls `strategy.emit_signal()` correctly
- [ ] Fetches market data for ticker
- [ ] Processes through `oms.process_signal()`
- [ ] Handles order rejection gracefully
- [ ] Logs all steps appropriately

**Portfolio Monitoring:**
- [ ] `get_portfolio_status()` fetches current prices
- [ ] Returns portfolio summary dict
- [ ] `print_portfolio_summary()` displays formatted output
- [ ] Shows all open positions with current prices

**Error Handling:**
- [ ] Handles missing historical data
- [ ] Handles indicator calculation errors
- [ ] Handles signal generation errors
- [ ] Handles market data fetch failures
- [ ] Continues cycle even if one ticker fails

**Test These:**
```python
from quant_strategies.execution import Portfolio, Ledger, BrokerAgent, PositionSizer, OrderManagementSystem
from quant_strategies.data.live_data_fetcher import LiveDataFetcher
from quant_strategies.strategies import BollingerRSIStrategy
from quant_strategies.strategies.live_strategy_runner import LiveStrategyRunner

# Setup
portfolio = Portfolio(100000)
ledger = Ledger("data/test.db")
oms = OrderManagementSystem(portfolio, ledger, BrokerAgent(), PositionSizer())
fetcher = LiveDataFetcher()
strategy = BollingerRSIStrategy()

# Create runner
runner = LiveStrategyRunner(strategy, oms, fetcher, ['AAPL', 'MSFT'])

# Test cycle
results = runner.run_cycle()
print(results)  # Should show signals/orders

# Test portfolio status
runner.print_portfolio_summary()
```

**Review Notes:**
```
[Add your comments here]




```

**Status:** [ ] Approved [ ] Needs Changes [ ] Questions

---

### 3. Live Trading Example (`scripts/live_trading_example.py` - 340 lines)

**Core Requirements:**
- [ ] `setup_logging()` configures logging to file and console
- [ ] `load_config()` loads YAML configuration
- [ ] `initialize_oms()` creates all OMS components
- [ ] `initialize_strategy()` loads strategy with parameters
- [ ] `run_live_trading()` orchestrates complete workflow

**Command Line Arguments:**
- [ ] `--config` specifies config file path
- [ ] `--cycles` limits number of trading cycles
- [ ] `--interval` overrides cycle interval in seconds
- [ ] Default behavior: infinite cycles until Ctrl+C
- [ ] Help text explains usage and examples

**Initialization Sequence:**
- [ ] Loads configuration from YAML
- [ ] Sets up logging as configured
- [ ] Creates Portfolio with initial cash
- [ ] Creates Ledger with database path
- [ ] Creates BrokerAgent with fees/slippage
- [ ] Creates PositionSizer
- [ ] Assembles OMS from components
- [ ] Initializes strategy with parameters
- [ ] Creates LiveDataFetcher with cache settings
- [ ] Displays watchlist with company names
- [ ] Creates LiveStrategyRunner

**Trading Loop:**
- [ ] Prints cycle number and timestamp
- [ ] Calls `runner.run_cycle()`
- [ ] Displays cycle results (signals/orders/errors)
- [ ] Prints portfolio summary after each cycle
- [ ] Sleeps for configured interval
- [ ] Handles KeyboardInterrupt gracefully
- [ ] Shows final summary when stopping

**Output Display:**
- [ ] Clear section headers with separators
- [ ] Shows initialization progress
- [ ] Displays watchlist with ticker names
- [ ] Shows cycle results
- [ ] Displays portfolio status
- [ ] Final summary at end

**Error Handling:**
- [ ] Checks config file exists
- [ ] Handles cycle errors without crashing
- [ ] Continues after individual ticker failures
- [ ] Logs errors appropriately
- [ ] Graceful shutdown on Ctrl+C

**Test Commands:**
```bash
# Test 1: Single cycle
python scripts/live_trading_example.py --cycles 1 --interval 1

# Test 2: Five cycles with 30s interval
python scripts/live_trading_example.py --cycles 5 --interval 30

# Test 3: Check help
python scripts/live_trading_example.py --help

# Test 4: Custom config
python scripts/live_trading_example.py --config my_config.yaml
```

**Review Notes:**
```
[Add your comments here]




```

**Status:** [ ] Approved [ ] Needs Changes [ ] Questions

---

### 4. Configuration File (`config/live_trading_config.yaml` - 54 lines)

**Watchlist Section:**
- [ ] Contains 10 liquid stock tickers
- [ ] Includes: AAPL, MSFT, GOOGL, TSLA, AMZN
- [ ] Includes: META, NVDA, JPM, V, WMT
- [ ] All tickers are valid and tradeable

**Schedule Section:**
- [ ] `trading_interval_minutes` set to reasonable value (5)
- [ ] `market_hours_only` flag present
- [ ] `start_time` and `end_time` configured (09:30-16:00 ET)
- [ ] `timezone` specified as America/New_York

**Data Section:**
- [ ] `historical_period` set (1y)
- [ ] `bars_for_indicators` configured (100)
- [ ] `cache_duration_seconds` reasonable (60s)
- [ ] `data_interval` specified (1d for daily)

**Portfolio Section:**
- [ ] `initial_cash` set appropriately (100000)
- [ ] `max_positions` limit defined (10)

**Strategy Section:**
- [ ] Strategy name specified (BollingerRSI)
- [ ] All required parameters included:
  - [ ] bb_period: 20
  - [ ] bb_std: 2
  - [ ] rsi_period: 14
  - [ ] rsi_oversold: 30
  - [ ] rsi_overbought: 70
  - [ ] volume_multiplier: 1.5

**Risk Management:**
- [ ] `max_position_size_pct` reasonable (0.10 = 10%)
- [ ] `min_cash_buffer_pct` reasonable (0.20 = 20%)

**Execution Settings:**
- [ ] `commission_per_share` specified (0.005)
- [ ] `min_commission` set (1.00)
- [ ] `slippage_bps` configured (5.0)

**Logging Settings:**
- [ ] Log level appropriate (INFO)
- [ ] Log file path specified
- [ ] Console logging enabled

**Test Loading:**
```python
import yaml

with open('config/live_trading_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Watchlist: {config['watchlist']}")
print(f"Initial cash: ${config['portfolio']['initial_cash']:,}")
print(f"Strategy: {config['strategy']['name']}")
```

**Review Notes:**
```
[Add your comments here]




```

**Status:** [ ] Approved [ ] Needs Changes [ ] Questions

---

## 📊 Integration Testing

### End-to-End Workflow Test

**Prerequisites:**
- [ ] OMS components working (from previous review)
- [ ] yfinance package installed
- [ ] Configuration file present

**Test 1: Single Cycle (Quick Test)**
```bash
python scripts/live_trading_example.py --cycles 1 --interval 1
```

**Expected Results:**
- [ ] System initializes without errors
- [ ] Fetches data for all watchlist tickers
- [ ] Calculates indicators successfully
- [ ] Generates signals (may be all HOLD if no setup)
- [ ] Displays portfolio summary
- [ ] No crashes or exceptions

**Test 2: Multiple Cycles**
```bash
python scripts/live_trading_example.py --cycles 3 --interval 10
```

**Expected Results:**
- [ ] Runs 3 complete cycles
- [ ] Waits 10 seconds between cycles
- [ ] Portfolio state persists across cycles
- [ ] Duplicate signals are prevented
- [ ] Final summary shows correct state

**Test 3: Data Fetching**
```python
from quant_strategies.data.live_data_fetcher import LiveDataFetcher

fetcher = LiveDataFetcher()

# Test multiple tickers
prices = fetcher.get_current_prices(['AAPL', 'MSFT', 'GOOGL'])
print(f"Fetched {len(prices)} prices")

# Test historical data
hist = fetcher.get_latest_bars('AAPL', n=100)
print(f"AAPL history: {len(hist)} bars")
print(hist.tail())
```

**Expected Results:**
- [ ] Fetches prices for all tickers (or shows warnings)
- [ ] Historical data has correct columns (open, high, low, close, volume)
- [ ] Data is recent (not stale)
- [ ] No exceptions thrown

**Test 4: Signal Generation**
```python
from quant_strategies.strategies import BollingerRSIStrategy
from quant_strategies.data.live_data_fetcher import LiveDataFetcher

strategy = BollingerRSIStrategy()
fetcher = LiveDataFetcher()

# Get data and test strategy
data = fetcher.get_latest_bars('AAPL', n=100)
if data is not None:
    indicators = strategy.calculate_indicators(data)
    signals = strategy.generate_signals(indicators)
    print(f"Latest signal: {signals.iloc[-1]['signal']}")
```

**Expected Results:**
- [ ] Indicators calculate without errors
- [ ] Signals generate (BUY/SELL/HOLD)
- [ ] Signal column present in output
- [ ] No NaN values in critical columns

**Test 5: OMS Integration**
- [ ] Strategy runner creates signals correctly
- [ ] Signals convert to OMS format
- [ ] OMS accepts and processes signals
- [ ] Orders execute through broker agent
- [ ] Portfolio updates correctly
- [ ] Ledger records all activity

**Test 6: Ledger Verification**
```python
from quant_strategies.execution import Ledger

ledger = Ledger("data/live_trading_ledger.db")

# Check what's recorded
signals = ledger.get_all_signals()
print(f"Total signals: {len(signals)}")

# View ledger contents
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/live_trading_ledger.db")
signals_df = pd.read_sql("SELECT * FROM signals", conn)
orders_df = pd.read_sql("SELECT * FROM orders", conn)
print(f"Signals in DB: {len(signals_df)}")
print(f"Orders in DB: {len(orders_df)}")
conn.close()
```

**Expected Results:**
- [ ] Ledger database created
- [ ] Signals recorded to database
- [ ] Orders recorded (if any generated)
- [ ] Can query ledger successfully
- [ ] Audit trail is complete

---

## 🔍 Data Quality Checks

### yfinance Data Validation

**Real-Time Prices:**
- [ ] Prices are current (within market hours or last close)
- [ ] Bid/ask spreads reasonable (<1% for liquid stocks)
- [ ] Volume data present
- [ ] No NaN or zero prices

**Historical Data:**
- [ ] Covers requested period (1y default)
- [ ] Has all OHLCV columns
- [ ] No large gaps in data
- [ ] Timezone handling correct
- [ ] Date range appropriate

**Cache Behavior:**
- [ ] First fetch takes longer (actual API call)
- [ ] Subsequent fetches faster (cached)
- [ ] Cache expires after timeout (60s)
- [ ] Stale data not used

---

## 🚨 Common Issues to Check

### Data Fetching Issues
- [ ] API rate limits not exceeded
- [ ] Network errors handled gracefully
- [ ] Market closed times handled (shows last close)
- [ ] Invalid tickers caught and logged

### Strategy Issues  
- [ ] Sufficient historical data for indicators (need 100+ bars)
- [ ] NaN handling in calculations
- [ ] Signal types converted correctly
- [ ] Duplicate signals prevented

### OMS Integration Issues
- [ ] Market data format matches OMS expectations
- [ ] Signal emission works from strategy
- [ ] Order sizing correct
- [ ] Portfolio constraints enforced

### Performance Issues
- [ ] Not too many API calls (use cache)
- [ ] Reasonable cycle times (<30s for 10 tickers)
- [ ] Memory usage stable over time
- [ ] No resource leaks

---

## 💡 Suggested Improvements

### Must Have Before Merge:
```
1. [Add critical issues here]


```

### Nice to Have:
```
1. [Add enhancement ideas here]


```

### Future Enhancements:
```
1. Market hours detection (only trade 9:30-4pm ET)
2. Multiple strategies running concurrently
3. Email/SMS alerts for trades
4. Better error recovery and retries
5. Save historical data locally to reduce API calls
```

---

## ✅ Final Review

### Overall Assessment
- [ ] Data fetcher works reliably
- [ ] Strategy runner integrates properly
- [ ] Live trading script runs smoothly
- [ ] Configuration is sensible
- [ ] Error handling is robust
- [ ] Logging is adequate
- [ ] Ready for paper trading

### Integration with OMS
- [ ] Signals flow correctly to OMS
- [ ] Orders execute properly
- [ ] Portfolio tracks positions
- [ ] Ledger records everything
- [ ] P&L calculations correct

### Code Quality
- [ ] No linter errors
- [ ] Docstrings present
- [ ] Error handling comprehensive
- [ ] Logging appropriate
- [ ] Configuration flexible

### Approval Status
- [ ] **APPROVED** - Ready to merge
- [ ] **APPROVED WITH MINOR CHANGES** - Small fixes needed
- [ ] **NEEDS REVISION** - Major changes required
- [ ] **BLOCKED** - Critical issues found

### Sign-off
```
Reviewer: _________________
Date: _____________________
Signature: ________________
```

---

## 📝 Action Items

### Before Merge:
1. [ ] _______________________________
2. [ ] _______________________________
3. [ ] _______________________________

### After Merge:
1. [ ] Test with live market data during trading hours
2. [ ] Monitor for 1-2 days to ensure stability
3. [ ] Build dashboard for visualization
4. [ ] _______________________________

### Dashboard Development (Next Phase):
1. [ ] Choose framework (Streamlit recommended)
2. [ ] Real-time portfolio display
3. [ ] P&L charts over time
4. [ ] Position details with live prices
5. [ ] Trade history from ledger
6. [ ] Performance metrics

