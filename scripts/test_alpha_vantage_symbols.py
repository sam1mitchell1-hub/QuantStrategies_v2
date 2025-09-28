#!/usr/bin/env python3
"""
Test Alpha Vantage symbols for UK indices.

This script tests different symbol variations to find the correct
FTSE 100 symbol for Alpha Vantage.
"""

import requests
import time
import os

def test_symbol(api_key, symbol):
    """Test a specific symbol with Alpha Vantage."""
    print(f"Testing symbol: {symbol}")
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'compact'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            print(f"  ❌ Error: {data['Error Message']}")
            return False
        elif 'Note' in data:
            print(f"  ⚠️  Note: {data['Note']}")
            return False
        elif 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            latest_date = max(time_series.keys())
            latest_price = time_series[latest_date]['4. close']
            print(f"  ✅ Success! Latest price: {latest_price} (Date: {latest_date})")
            return True
        else:
            print(f"  ❌ Unexpected response: {list(data.keys())}")
            return False
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def main():
    """Test various UK index symbols."""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        print("❌ ALPHA_VANTAGE_API_KEY not found")
        return
    
    print(f"Using API key: {api_key[:8]}...")
    print("Testing UK index symbols...")
    print("=" * 50)
    
    # Test various UK index symbols
    symbols_to_test = [
        "FTSE",      # Original attempt
        "^FTSE",     # Yahoo Finance style
        "FTSE.L",    # London exchange
        "UKX",       # Intrinio style
        "UKX.L",     # UKX on London
        "FTSE100",   # Alternative naming
        "FTSE100.L", # FTSE 100 on London
        "FTSE-100",  # With dash
        "FTSE_100",  # With underscore
        "UKX100",    # Alternative
        "LSE:FTSE",  # Exchange prefix
        "LON:FTSE",  # London prefix
    ]
    
    working_symbols = []
    
    for symbol in symbols_to_test:
        success = test_symbol(api_key, symbol)
        if success:
            working_symbols.append(symbol)
        
        # Rate limiting - wait 12 seconds between calls
        print("  Waiting 12 seconds for rate limiting...")
        time.sleep(12)
        print()
    
    print("=" * 50)
    print("RESULTS:")
    if working_symbols:
        print(f"✅ Working symbols: {working_symbols}")
    else:
        print("❌ No working symbols found")
        print("\nThis suggests Alpha Vantage may not support FTSE 100 data.")
        print("Consider using alternative data sources:")
        print("1. Yahoo Finance (free)")
        print("2. Eurex Exchange (professional)")
        print("3. ICE Futures Europe (professional)")
        print("4. Interactive Brokers (if you have account)")

if __name__ == "__main__":
    main()
