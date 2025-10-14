# FTSE 100 Data Provider Setup Guide

## Overview

The FTSE 100 data provider allows you to fetch real-time and historical data for the FTSE 100 index and its options chain using the Intrinio API. This guide will help you set up the necessary API keys and configure the data provider.

## Prerequisites

1. **Intrinio API Account**: You need an active Intrinio API account
2. **Python Dependencies**: The required packages are already included in the project

## Setup Steps

### 1. Get Intrinio API Key

1. Visit [Intrinio](https://intrinio.com/) and create an account
2. Navigate to your account settings and find your API key
3. Copy the API key for the next step

### 2. Set Environment Variables

#### Option A: Environment Variable (Recommended)

```bash
# Set your API key
export INTRINIO_API_KEY="your_api_key_here"

# Optional: Set other configuration
export DEFAULT_DATA_PROVIDER="intrinio"
export FTSE_SYMBOL="UKX"
export DEFAULT_TIMEZONE="Europe/London"
```

#### Option B: .env File

1. Copy the example environment file:
   ```bash
   cp config/env.example .env
   ```

2. Edit `.env` and add your API key:
   ```bash
   # Edit .env file
   nano .env
   
   # Add your API key
   INTRINIO_API_KEY=your_api_key_here
   ```

#### Option C: Python Environment

```python
import os
os.environ['INTRINIO_API_KEY'] = 'your_api_key_here'
```

### 3. Test the Setup

Run the test script to verify everything is working:

```bash
python scripts/test_ftse_data.py
```

Expected output:
```
FTSE 100 Data Provider Test
==================================================
Using API key: abc12345...

=== Testing Intrinio API Connection ===
✅ API connection successful!

=== Testing Index Snapshot ===
Fetching FTSE 100 snapshot for 2024-01-15...
✅ Index snapshot retrieved:
   Timestamp: 2024-01-15 15:45:00+00:00
   Index Price: 7654.32
   Volume: 1234567

=== Testing Options Chain ===
Fetching FTSE 100 options chain for 2024-01-15...
✅ Options chain retrieved:
   Total options: 245
   Calls: 123
   Puts: 122
   Unique strikes: 15
   Unique expiries: 4

=== Test Results ===
Passed: 4/4 tests
🎉 All tests passed! FTSE data provider is working correctly.
```

## Usage Examples

### Basic Usage

```python
from data_providers.intrinio_ftse import get_index_snapshot, get_option_chain_snapshot

# Get index snapshot
snapshot = get_index_snapshot(
    date="2024-01-15",
    near_time="15:45",
    window=20
)
print(f"FTSE 100 at {snapshot.timestamp}: {snapshot.index_px}")

# Get options chain
options_df = get_option_chain_snapshot(
    date="2024-01-15",
    near_time="15:45",
    window=20,
    maturity_bounds=(7, 60),
    spread_limit=0.5,
    min_oi=100
)
print(f"Found {len(options_df)} options")
```

### Advanced Usage

```python
from data_providers.intrinio_ftse import IntrinioFTSEProvider

# Create provider instance
provider = IntrinioFTSEProvider()

# Test connection
if provider.test_connection():
    print("API connection successful")
    
    # Get index data
    snapshot = provider.get_index_snapshot(
        date="2024-01-15",
        near_time="15:45",
        window=20
    )
    
    # Get options data
    options_df = provider.get_option_chain_snapshot(
        date="2024-01-15",
        near_time="15:45",
        window=20,
        maturity_bounds=(7, 60),
        spread_limit=0.5,
        min_oi=100
    )
```

## Data Format

### Index Snapshot

```python
@dataclass
class IndexSnapshot:
    timestamp: pd.Timestamp  # London timezone
    index_px: float         # Index price
    volume: Optional[int]   # Volume (if available)
```

### Options Chain DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | pd.Timestamp | Snapshot timestamp (London timezone) |
| `cp` | str | Call/Put indicator |
| `strike` | float | Strike price |
| `expiry` | pd.Timestamp | Expiration date |
| `bid` | float | Bid price |
| `ask` | float | Ask price |
| `mid` | float | Mid price (bid+ask)/2 |
| `oi` | int | Open interest |
| `volume` | int | Volume (if available) |
| `iv` | float | Implied volatility (if available) |
| `delta` | float | Delta (if available) |
| `gamma` | float | Gamma (if available) |
| `theta` | float | Theta (if available) |
| `vega` | float | Vega (if available) |

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INTRINIO_API_KEY` | Required | Your Intrinio API key |
| `DEFAULT_DATA_PROVIDER` | "intrinio" | Default data provider |
| `FTSE_SYMBOL` | "UKX" | FTSE 100 symbol |
| `DEFAULT_TIMEZONE` | "Europe/London" | Default timezone |
| `LOG_LEVEL` | "INFO" | Logging level |

### Function Parameters

#### `get_index_snapshot()`

- `date`: Date in YYYY-MM-DD format
- `near_time`: Target time in HH:MM format (default: "15:45")
- `window`: Window in minutes around target time (default: 20)

#### `get_option_chain_snapshot()`

- `date`: Date in YYYY-MM-DD format
- `near_time`: Target time in HH:MM format (default: "15:45")
- `window`: Window in minutes around target time (default: 20)
- `maturity_bounds`: (min_days, max_days) for option maturity (default: (7, 60))
- `spread_limit`: Maximum relative spread (ask-bid)/mid (default: 0.5)
- `min_oi`: Minimum open interest (default: 100)

## Integration with Strategy Framework

The FTSE data provider can be easily integrated with the existing strategy framework:

```python
from data_providers.intrinio_ftse import IntrinioFTSEProvider
from strategy import FeatureBuilder, StrategyConfig

# Load configuration
config = StrategyConfig.from_yaml('config/strategy_config.yaml')

# Create data provider
ftse_provider = IntrinioFTSEProvider()

# Fetch FTSE data
index_snapshot = ftse_provider.get_index_snapshot("2024-01-15")
options_df = ftse_provider.get_option_chain_snapshot("2024-01-15")

# Convert to strategy format
market_data = {
    'prices': create_price_dataframe(index_snapshot),
    'options': options_df
}

# Build features
feature_builder = FeatureBuilder(config)
feature_set = feature_builder.build_features(**market_data)
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   ValueError: INTRINIO_API_KEY not found in environment variables
   ```
   **Solution**: Set the `INTRINIO_API_KEY` environment variable

2. **No Data Found**
   ```
   ValueError: No intraday data found for 2024-01-15 in time window 15:45±20m
   ```
   **Solution**: Check if the date is a trading day and within market hours

3. **API Rate Limits**
   ```
   requests.RequestException: 429 Too Many Requests
   ```
   **Solution**: Wait before making more requests or upgrade your API plan

4. **Invalid Date Format**
   ```
   ValueError: time data '15:45' does not match format '%H:%M'
   ```
   **Solution**: Use HH:MM format for time (e.g., "15:45", "09:30")

### Debug Mode

Enable debug logging to see detailed API requests:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

### Testing Connection

```python
from data_providers.intrinio_ftse import IntrinioFTSEProvider

provider = IntrinioFTSEProvider()
if provider.test_connection():
    print("Connection successful!")
else:
    print("Connection failed!")
```

## API Limits and Costs

- **Free Tier**: Limited requests per month
- **Paid Plans**: Higher limits and additional features
- **Rate Limits**: Vary by plan (check Intrinio documentation)

## Support

- **Intrinio Documentation**: [https://intrinio.com/developers](https://intrinio.com/developers)
- **API Status**: [https://status.intrinio.com/](https://status.intrinio.com/)
- **Support**: Contact Intrinio support for API-related issues

## Next Steps

1. Set up your API key
2. Run the test script
3. Integrate with your strategy framework
4. Start fetching real FTSE 100 data for backtesting and live trading
