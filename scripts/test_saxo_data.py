#!/usr/bin/env python3
"""
Test script for Saxo Bank data provider.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.saxo_data_provider import SaxoDataProvider
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_saxo_connection():
    """Test the Saxo API connection."""
    print("Testing Saxo Bank API connection...")
    
    try:
        provider = SaxoDataProvider()
        
        # Test connection
        if provider.test_connection():
            print("✅ Saxo API connection successful!")
            return provider
        else:
            print("❌ Saxo API connection failed!")
            return None
    except Exception as e:
        print(f"❌ Error initializing Saxo provider: {e}")
        return None


def test_instrument_search(provider: SaxoDataProvider):
    """Test instrument search functionality."""
    print("\nTesting instrument search...")
    
    test_symbols = ["Microsoft", "Tesla", "Apple", "Google"]
    
    for symbol in test_symbols:
        print(f"\nSearching for: {symbol}")
        instruments = provider.search_instruments(symbol, "Stock")
        
        if instruments:
            print(f"Found {len(instruments)} instruments:")
            for i, instrument in enumerate(instruments[:3]):  # Show first 3
                print(f"  {i+1}. {instrument.get('Symbol', 'N/A')} - {instrument.get('Description', 'N/A')} (UIC: {instrument.get('Uic', 'N/A')})")
        else:
            print("No instruments found")


def test_historical_data(provider: SaxoDataProvider):
    """Test historical data fetching."""
    print("\nTesting historical data fetching...")
    
    # Test with a few major stocks
    symbols = ["MSFT", "TSLA", "AAPL"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        try:
            data = provider.get_equity_data(symbol, start_date, end_date, "1d")
            
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched {len(data)} data points for {symbol}")
                print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Latest close: ${data['close'].iloc[-1]:.2f}")
            else:
                print(f"❌ No data received for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")


def test_multiple_equities(provider: SaxoDataProvider):
    """Test fetching data for multiple equities at once."""
    print("\nTesting multiple equities fetch...")
    
    symbols = ["MSFT", "TSLA", "AAPL", "GOOGL"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    results = provider.get_multiple_equities(symbols, start_date, end_date, "1d")
    
    print(f"\nResults summary:")
    for symbol, data in results.items():
        if data is not None and not data.empty:
            print(f"  {symbol}: {len(data)} data points, latest close: ${data['close'].iloc[-1]:.2f}")
        else:
            print(f"  {symbol}: No data")


def main():
    """Main test function."""
    print("Saxo Bank Data Provider Test")
    print("=" * 40)
    
    # Test connection
    provider = test_saxo_connection()
    if not provider:
        print("\n❌ Cannot proceed without API connection. Please check your API key in saxo_config.json")
        return
    
    # Test instrument search
    test_instrument_search(provider)
    
    # Test historical data
    test_historical_data(provider)
    
    # Test multiple equities
    test_multiple_equities(provider)
    
    print("\n" + "=" * 40)
    print("Test completed!")


if __name__ == "__main__":
    main()
