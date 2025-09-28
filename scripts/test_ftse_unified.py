#!/usr/bin/env python3
"""
Test script for unified FTSE 100 data manager.

This script tests the unified data manager that automatically
selects the best available data provider.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from data_providers.ftse_data_manager import FTSEDataManager, get_ftse_index_snapshot, get_ftse_historical_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_manager():
    """Test the unified data manager."""
    print("=== Testing Unified FTSE Data Manager ===")
    
    try:
        # Initialize manager
        manager = FTSEDataManager()
        
        # Check available providers
        available = manager.get_available_providers()
        print(f"Available providers: {available}")
        
        # Check working provider
        working = manager.get_working_provider()
        if working:
            print(f"Working provider: {working}")
            return True
        else:
            print("❌ No working providers found")
            return False
            
    except Exception as e:
        print(f"❌ Data manager error: {e}")
        return False


def test_index_snapshot():
    """Test index snapshot functionality."""
    print("\n=== Testing Index Snapshot ===")
    
    try:
        # Test with yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching FTSE 100 snapshot for {yesterday}...")
        snapshot = get_ftse_index_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20
        )
        
        print(f"✅ Index snapshot retrieved:")
        print(f"   Timestamp: {snapshot.timestamp}")
        print(f"   Index Price: {snapshot.index_px:.2f}")
        print(f"   Source: {snapshot.source}")
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
        data = get_ftse_historical_data(
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
        if 'Volume' in data.columns:
            print(f"   Average volume: {data['Volume'].mean():,.0f}")
        
        # Show sample data
        print("\n   Sample data (last 5 days):")
        sample_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in data.columns:
            sample_cols.append('Volume')
        print(data[sample_cols].tail().to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Historical data error: {e}")
        return False


def test_provider_fallback():
    """Test provider fallback functionality."""
    print("\n=== Testing Provider Fallback ===")
    
    try:
        # Test with different provider preferences
        providers_to_test = ["intrinio", "yahoo", "auto"]
        
        for provider in providers_to_test:
            print(f"\nTesting with provider preference: {provider}")
            try:
                manager = FTSEDataManager(provider)
                working = manager.get_working_provider()
                if working:
                    print(f"   ✅ Working provider: {working}")
                else:
                    print(f"   ❌ No working provider found")
            except Exception as e:
                print(f"   ❌ Error with {provider}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider fallback error: {e}")
        return False


def test_data_quality():
    """Test data quality."""
    print("\n=== Testing Data Quality ===")
    
    try:
        # Get historical data for analysis
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        
        data = get_ftse_historical_data(
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
    print("Unified FTSE 100 Data Manager Test")
    print("=" * 50)
    print("This test will automatically select the best available data provider")
    print("(Intrinio if available and working, otherwise Yahoo Finance)")
    print()
    
    # Run tests
    tests = [
        test_data_manager,
        test_index_snapshot,
        test_historical_data,
        test_provider_fallback,
        test_data_quality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All tests passed! Unified data manager is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
