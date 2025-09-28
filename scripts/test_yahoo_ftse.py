#!/usr/bin/env python3
"""
Test script for Yahoo Finance FTSE 100 data provider.

This script tests the Yahoo Finance fallback provider when Intrinio
is not available or doesn't have the required access level.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from data_providers.yahoo_ftse import YahooFTSEProvider, get_index_snapshot, get_historical_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection():
    """Test Yahoo Finance connection."""
    print("=== Testing Yahoo Finance Connection ===")
    
    try:
        provider = YahooFTSEProvider()
        if provider.test_connection():
            print("✅ Yahoo Finance connection successful!")
            return True
        else:
            print("❌ Yahoo Finance connection failed!")
            return False
    except Exception as e:
        print(f"❌ Yahoo Finance connection error: {e}")
        return False


def test_index_snapshot():
    """Test index snapshot functionality."""
    print("\n=== Testing Index Snapshot ===")
    
    try:
        # Test with yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching FTSE 100 snapshot for {yesterday}...")
        snapshot = get_index_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20
        )
        
        print(f"✅ Index snapshot retrieved:")
        print(f"   Timestamp: {snapshot.timestamp}")
        print(f"   Index Price: {snapshot.index_px:.2f}")
        if snapshot.volume:
            print(f"   Volume: {snapshot.volume:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Index snapshot error: {e}")
        return False


def test_historical_data():
    """Test historical data functionality."""
    print("\n=== Testing Historical Data ===")
    
    try:
        # Test with last 30 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        print(f"Fetching historical data from {start_date} to {end_date}...")
        data = get_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No historical data received")
            return False
        
        print(f"✅ Historical data retrieved:")
        print(f"   Records: {len(data)}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Latest close: {data['Close'].iloc[-1]:.2f}")
        print(f"   Average volume: {data['Volume'].mean():,.0f}")
        
        # Show sample data
        print("\n   Sample data (last 5 days):")
        sample_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        print(data[sample_cols].tail().to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Historical data error: {e}")
        return False


def test_options_data():
    """Test options data functionality."""
    print("\n=== Testing Options Data ===")
    
    try:
        # Test with yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching FTSE 100 options chain for {yesterday}...")
        provider = YahooFTSEProvider()
        options_df = provider.get_option_chain_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        if options_df.empty:
            print("⚠️  No options data available (expected with Yahoo Finance)")
            print("   Yahoo Finance doesn't provide FTSE 100 options data")
            print("   Consider using Intrinio or another provider for options")
            return True
        else:
            print(f"✅ Options data retrieved: {len(options_df)} options")
            return True
        
    except Exception as e:
        print(f"❌ Options data error: {e}")
        return False


def test_data_quality():
    """Test data quality."""
    print("\n=== Testing Data Quality ===")
    
    try:
        # Get historical data for analysis
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        
        data = get_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No data to analyze")
            return False
        
        print(f"   Total records: {len(data)}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Missing values: {data.isnull().sum().sum()}")
        
        # Price analysis
        print(f"   Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
        print(f"   Average close: {data['Close'].mean():.2f}")
        print(f"   Price volatility: {data['Close'].std():.2f}")
        
        # Volume analysis
        if 'Volume' in data.columns:
            print(f"   Average volume: {data['Volume'].mean():,.0f}")
            print(f"   Volume range: {data['Volume'].min():,} - {data['Volume'].max():,}")
        
        # Check for gaps
        date_diffs = data.index.to_series().diff().dt.days
        gaps = date_diffs[date_diffs > 1]
        if not gaps.empty:
            print(f"   Data gaps: {len(gaps)} gaps > 1 day")
        else:
            print("   Data continuity: No significant gaps")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality analysis error: {e}")
        return False


def main():
    """Main test function."""
    print("Yahoo Finance FTSE 100 Data Provider Test")
    print("=" * 50)
    print("Note: This is a fallback provider when Intrinio is not available")
    print("Yahoo Finance provides index data but not options data")
    print()
    
    # Run tests
    tests = [
        test_connection,
        test_index_snapshot,
        test_historical_data,
        test_options_data,
        test_data_quality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All tests passed! Yahoo Finance provider is working correctly.")
        print("\nNote: For options data, you'll need to use Intrinio or another provider.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
