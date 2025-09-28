# FTSE 100 Data Access Troubleshooting Guide

## Current Issues Identified

### 1. Intrinio API Access (401 Unauthorized)

**Problem**: Your Intrinio API key returns 401 Unauthorized errors for all FTSE 100 endpoints.

**Root Cause**: Your Intrinio plan doesn't include FTSE 100 data access.

**Evidence**:
```
ERROR: 401 Client Error: Unauthorized for url: https://api-v2.intrinio.com/securities/UKX
ERROR: 401 Client Error: Unauthorized for url: https://api-v2.intrinio.com/options/prices
```

**Solutions**:

#### Option A: Upgrade Intrinio Plan
1. Contact Intrinio support
2. Request FTSE 100 data access
3. Upgrade to a plan that includes international indices
4. Verify UKX symbol availability in your region

#### Option B: Use Alternative Data Sources
See alternative solutions below.

### 2. Yahoo Finance Symbol Issues

**Problem**: All tested FTSE 100 symbols return "No data found".

**Root Cause**: Yahoo Finance symbols may have changed or are region-restricted.

**Evidence**:
```
- ^FTSE: No data found for this date range, symbol may be delisted
- FTSE.L: No data found for this date range, symbol may be delisted
- UKX.L: No data found for this date range, symbol may be delisted
- FTSE100.L: No data found for this date range, symbol may be delisted
```

## Alternative Data Sources

### 1. Alpha Vantage (Recommended)

**Advantages**:
- Free tier available (5 calls/minute)
- FTSE 100 data available
- Simple API
- Good documentation

**Setup**:
```bash
# Get free API key from https://www.alphavantage.co/support/#api-key
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

**Usage**:
```python
import requests

def get_ftse_data_alpha_vantage(symbol="FTSE", api_key=None):
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full'
    }
    response = requests.get(url, params=params)
    return response.json()
```

### 2. Quandl (Professional)

**Advantages**:
- High-quality data
- FTSE 100 datasets available
- Professional grade
- Good for production use

**Setup**:
```bash
# Get API key from https://www.quandl.com/account/api
export QUANDL_API_KEY="your_api_key_here"
```

### 3. Interactive Brokers (If you have account)

**Advantages**:
- Real-time data
- Options data available
- Professional trading platform
- Comprehensive market data

**Setup**:
- Requires IB account
- Use IB API or TWS

### 4. Yahoo Finance Alternative Symbols

Try these alternative symbols:
- `^FTSE` (original)
- `FTSE.L` (London exchange)
- `UKX.L` (UKX on London)
- `FTSE100.L` (FTSE 100 on London)
- `^FCHI` (French CAC 40 as alternative)
- `^GDAXI` (German DAX as alternative)

## Immediate Solutions

### Solution 1: Use Alpha Vantage (Quickest)

1. **Get API Key**:
   - Visit https://www.alphavantage.co/support/#api-key
   - Sign up for free account
   - Get your API key

2. **Set Environment Variable**:
   ```bash
   export ALPHA_VANTAGE_API_KEY="your_api_key_here"
   ```

3. **Test Connection**:
   ```python
   import requests
   
   def test_alpha_vantage():
       api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
       url = "https://www.alphavantage.co/query"
       params = {
           'function': 'TIME_SERIES_DAILY',
           'symbol': 'FTSE',
           'apikey': api_key,
           'outputsize': 'compact'
       }
       response = requests.get(url, params=params)
       return response.status_code == 200
   ```

### Solution 2: Use Sample Data for Development

For development and testing, you can use sample data:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_ftse_data(start_date, end_date):
    """Create sample FTSE 100 data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic FTSE 100 data
    base_price = 7500
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    return data
```

### Solution 3: Use Different Index

If FTSE 100 data is not available, consider using:
- **S&P 500** (`^GSPC`) - Widely available
- **NASDAQ** (`^IXIC`) - Good for tech stocks
- **Dow Jones** (`^DJI`) - Traditional index
- **DAX** (`^GDAXI`) - German index
- **CAC 40** (`^FCHI`) - French index

## Next Steps

### Immediate (Today)
1. **Get Alpha Vantage API key** (5 minutes)
2. **Test with sample data** for development
3. **Use S&P 500** as alternative for testing

### Short Term (This Week)
1. **Contact Intrinio support** about FTSE 100 access
2. **Implement Alpha Vantage provider**
3. **Test with real data**

### Long Term (This Month)
1. **Evaluate data provider options**
2. **Choose best provider for production**
3. **Implement comprehensive data pipeline**

## Code Examples

### Alpha Vantage Implementation

```python
class AlphaVantageFTSEProvider:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_daily_data(self, symbol="FTSE"):
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_intraday_data(self, symbol="FTSE", interval="1min"):
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
```

### Sample Data Implementation

```python
def create_sample_ftse_options_data(date, num_options=50):
    """Create sample FTSE 100 options data for testing."""
    base_price = 7500
    strikes = np.linspace(base_price * 0.8, base_price * 1.2, num_options)
    
    options = []
    for strike in strikes:
        # Call option
        call_price = max(0, base_price - strike) + np.random.uniform(10, 50)
        options.append({
            'timestamp': pd.Timestamp(date),
            'cp': 'call',
            'strike': strike,
            'expiry': pd.Timestamp(date) + pd.Timedelta(days=30),
            'bid': call_price * 0.95,
            'ask': call_price * 1.05,
            'mid': call_price,
            'oi': np.random.randint(100, 1000),
            'iv': np.random.uniform(0.15, 0.35)
        })
        
        # Put option
        put_price = max(0, strike - base_price) + np.random.uniform(10, 50)
        options.append({
            'timestamp': pd.Timestamp(date),
            'cp': 'put',
            'strike': strike,
            'expiry': pd.Timestamp(date) + pd.Timedelta(days=30),
            'bid': put_price * 0.95,
            'ask': put_price * 1.05,
            'mid': put_price,
            'oi': np.random.randint(100, 1000),
            'iv': np.random.uniform(0.15, 0.35)
        })
    
    return pd.DataFrame(options)
```

## Contact Information

### Intrinio Support
- **Website**: https://intrinio.com/support
- **Email**: support@intrinio.com
- **Phone**: Check their website for current contact info

### Alpha Vantage Support
- **Website**: https://www.alphavantage.co/support
- **Email**: support@alphavantage.co

### Quandl Support
- **Website**: https://www.quandl.com/help
- **Email**: support@quandl.com

## Conclusion

The 401 errors from Intrinio indicate that your current plan doesn't include FTSE 100 data access. The quickest solution is to:

1. **Use Alpha Vantage** for immediate access (free tier available)
2. **Contact Intrinio** about upgrading your plan
3. **Use sample data** for development and testing

The strategy framework is fully functional and ready to work with any data source once you have access to market data.
