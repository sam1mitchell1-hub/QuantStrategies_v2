# Saxo Bank API Integration

This document describes how to use the Saxo Bank API integration for fetching historical equity data.

## Setup

### 1. API Key Configuration

1. Copy your 24-hour API key from your Saxo Bank account
2. Open `data_providers/saxo_config.json`
3. Replace `YOUR_24HR_API_KEY_HERE` with your actual API key:

```json
{
  "api_key": "your_actual_api_key_here",
  "base_url": "https://gateway.saxobank.com/sim/openapi",
  "timeout": 30,
  "retry_attempts": 3
}
```

### 2. Install Dependencies

The Saxo integration uses the `requests` library which should already be installed. If not:

```bash
pip install requests
```

## Usage

### Basic Usage

```python
from data_providers.saxo_data_provider import SaxoDataProvider

# Initialize provider
provider = SaxoDataProvider()

# Test connection
if provider.test_connection():
    print("Connected to Saxo API!")

# Search for instruments
instruments = provider.search_instruments("Microsoft", "Stock")
print(f"Found {len(instruments)} Microsoft instruments")

# Get historical data for a specific symbol
data = provider.get_equity_data("MSFT", "2024-01-01", "2024-12-31", "1d")
print(f"Retrieved {len(data)} data points")
```

### Command Line Scripts

#### Test Connection
```bash
python scripts/test_saxo_data.py
```

#### Fetch Historical Data
```bash
# Fetch data for specific symbols
python scripts/fetch_saxo_historical_data.py --symbols AAPL,MSFT,TSLA --start 2024-01-01 --end 2024-12-31

# Fetch data for major liquid equities
python scripts/fetch_saxo_historical_data.py --major-only --start 2024-01-01 --interval 1d

# Fetch intraday data
python scripts/fetch_saxo_historical_data.py --symbols AAPL --start 2024-12-01 --interval 1h
```

## Available Methods

### SaxoDataProvider Class

- `test_connection()` - Test API connectivity
- `search_instruments(query, asset_type)` - Search for instruments by name/symbol
- `get_instrument_details(uic)` - Get detailed instrument information
- `get_historical_data(uic, start_date, end_date, interval)` - Get historical data by UIC
- `get_equity_data(symbol, start_date, end_date, interval)` - Get historical data by symbol
- `get_multiple_equities(symbols, start_date, end_date, interval)` - Get data for multiple symbols

### Supported Intervals

- `1m` - 1 minute
- `5m` - 5 minutes  
- `1h` - 1 hour
- `1d` - 1 day (default)
- `1w` - 1 week
- `1M` - 1 month

### Data Format

The returned DataFrame contains the following columns:
- `timestamp` - DateTime index
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume

## Major Liquid Equities

The integration includes a predefined list of major liquid equities suitable for volatility modeling:

**Technology**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX, ADBE, CRM, PYPL, INTC, AMD, ORCL, CSCO, IBM

**Financial**: JPM, BAC, V, MA

**Consumer/Healthcare**: WMT, JNJ, PG, UNH, HD, DIS

**ETFs/Indices**: SPY, QQQ, IWM, VIX

## Error Handling

The provider includes robust error handling:
- Automatic retry with exponential backoff
- Rate limiting protection
- Detailed logging for debugging
- Graceful handling of missing data

## Rate Limits

Saxo Bank API has rate limits. The provider includes:
- 2-second delays between requests
- Configurable retry attempts
- Request timeout settings

## Troubleshooting

### Common Issues

1. **"No data received"** - Check if the symbol exists and is tradeable
2. **"API connection failed"** - Verify your API key is correct and active
3. **"No instruments found"** - Try different search terms or check asset type

### Debug Mode

Enable debug logging to see detailed API interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Notes

- Never commit your API key to version control
- The `saxo_config.json` file is in `.gitignore` to prevent accidental commits
- Consider using environment variables for production deployments
