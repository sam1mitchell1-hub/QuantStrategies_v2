#!/usr/bin/env python3
"""
Test script for FTSE 100 data provider.

This script demonstrates how to use the Intrinio FTSE data provider
to fetch index and options data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from data_providers.intrinio_ftse import IntrinioFTSEProvider, get_index_snapshot, get_option_chain_snapshot
from data_providers.config import DataProviderConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection():
    """Test API connection."""
    print("=== Testing Intrinio API Connection ===")
    
    try:
        provider = IntrinioFTSEProvider()
        if provider.test_connection():
            print("✅ API connection successful!")
            return True
        else:
            print("❌ API connection failed!")
            return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
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


def test_options_chain():
    """Test options chain functionality."""
    print("\n=== Testing Options Chain ===")
    
    try:
        # Test with yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching FTSE 100 options chain for {yesterday}...")
        options_df = get_option_chain_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        if options_df.empty:
            print("⚠️  No options data found (this might be normal for some dates)")
            return True
        
        print(f"✅ Options chain retrieved:")
        print(f"   Total options: {len(options_df)}")
        print(f"   Calls: {len(options_df[options_df['cp'] == 'call'])}")
        print(f"   Puts: {len(options_df[options_df['cp'] == 'put'])}")
        print(f"   Unique strikes: {options_df['strike'].nunique()}")
        print(f"   Unique expiries: {options_df['expiry'].nunique()}")
        
        # Show sample data
        print("\n   Sample data:")
        sample_cols = ['cp', 'strike', 'expiry', 'bid', 'ask', 'mid', 'oi']
        print(options_df[sample_cols].head(10).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Options chain error: {e}")
        return False


def test_data_quality():
    """Test data quality and filtering."""
    print("\n=== Testing Data Quality ===")
    
    try:
        # Test with yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Analyzing data quality for {yesterday}...")
        options_df = get_option_chain_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        if options_df.empty:
            print("⚠️  No options data to analyze")
            return True
        
        # Data quality checks
        print(f"   Total records: {len(options_df)}")
        print(f"   Records with bid > 0: {len(options_df[options_df['bid'] > 0])}")
        print(f"   Records with ask > bid: {len(options_df[options_df['ask'] > options_df['bid']])}")
        print(f"   Records with OI >= 100: {len(options_df[options_df['oi'] >= 100])}")
        
        # Spread analysis
        options_df['rel_spread'] = (options_df['ask'] - options_df['bid']) / options_df['mid']
        print(f"   Average relative spread: {options_df['rel_spread'].mean():.3f}")
        print(f"   Max relative spread: {options_df['rel_spread'].max():.3f}")
        
        # Strike range
        print(f"   Strike range: {options_df['strike'].min():.0f} - {options_df['strike'].max():.0f}")
        
        # Maturity analysis
        options_df['days_to_expiry'] = (options_df['expiry'].dt.date - options_df['timestamp'].dt.date).dt.days
        print(f"   Maturity range: {options_df['days_to_expiry'].min()} - {options_df['days_to_expiry'].max()} days")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality analysis error: {e}")
        return False


def main():
    """Main test function."""
    print("FTSE 100 Data Provider Test")
    print("=" * 50)
    
    # Check configuration
    config = DataProviderConfig.from_env()
    if not config.intrinio_api_key:
        print("❌ INTRINIO_API_KEY not found in environment variables")
        print("   Please set your API key:")
        print("   export INTRINIO_API_KEY='your_api_key_here'")
        print("   or create a .env file with INTRINIO_API_KEY=your_api_key_here")
        return False
    
    print(f"Using API key: {config.intrinio_api_key[:8]}...")
    
    # Run tests
    tests = [
        test_connection,
        test_index_snapshot,
        test_options_chain,
        test_data_quality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All tests passed! FTSE data provider is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
