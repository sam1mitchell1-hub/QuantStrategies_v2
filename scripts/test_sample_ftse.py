#!/usr/bin/env python3
"""
Test script for sample FTSE 100 data provider.

This script tests the sample data provider for development and testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from data_providers.sample_ftse import SampleFTSEProvider, get_index_snapshot, get_historical_data, get_option_chain_snapshot
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection():
    """Test sample data generation."""
    print("=== Testing Sample Data Generation ===")
    
    try:
        provider = SampleFTSEProvider()
        if provider.test_connection():
            print("✅ Sample data generation successful!")
            return True
        else:
            print("❌ Sample data generation failed!")
            return False
    except Exception as e:
        print(f"❌ Sample data generation error: {e}")
        return False


def test_index_snapshot():
    """Test index snapshot functionality."""
    print("\n=== Testing Index Snapshot ===")
    
    try:
        # Test with yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Generating sample FTSE 100 snapshot for {yesterday}...")
        snapshot = get_index_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20
        )
        
        print(f"✅ Sample index snapshot generated:")
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
        
        print(f"Generating sample historical data from {start_date} to {end_date}...")
        data = get_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No historical data generated")
            return False
        
        print(f"✅ Sample historical data generated:")
        print(f"   Records: {len(data)}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Latest close: {data['close'].iloc[-1]:.2f}")
        print(f"   Average volume: {data['volume'].mean():,.0f}")
        
        # Show sample data
        print("\n   Sample data (last 5 days):")
        sample_cols = ['open', 'high', 'low', 'close', 'volume']
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
        
        print(f"Generating sample FTSE 100 options chain for {yesterday}...")
        options_df = get_option_chain_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        if options_df.empty:
            print("❌ No options data generated")
            return False
        
        print(f"✅ Sample options data generated:")
        print(f"   Total options: {len(options_df)}")
        print(f"   Calls: {len(options_df[options_df['cp'] == 'call'])}")
        print(f"   Puts: {len(options_df[options_df['cp'] == 'put'])}")
        print(f"   Unique strikes: {options_df['strike'].nunique()}")
        print(f"   Unique expiries: {options_df['expiry'].nunique()}")
        
        # Show sample data
        print("\n   Sample options data:")
        sample_cols = ['cp', 'strike', 'expiry', 'bid', 'ask', 'mid', 'oi', 'iv']
        print(options_df[sample_cols].head(10).to_string(index=False))
        
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
        print(f"   Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        print(f"   Average close: {data['close'].mean():.2f}")
        print(f"   Price volatility: {data['close'].std():.2f}")
        
        # Volume analysis
        if 'volume' in data.columns:
            print(f"   Average volume: {data['volume'].mean():,.0f}")
            print(f"   Volume range: {data['volume'].min():,} - {data['volume'].max():,}")
        
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


def test_options_quality():
    """Test options data quality."""
    print("\n=== Testing Options Data Quality ===")
    
    try:
        # Get options data for analysis
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        options_df = get_option_chain_snapshot(
            date=yesterday,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        if options_df.empty:
            print("❌ No options data to analyze")
            return False
        
        print(f"   Total options: {len(options_df)}")
        print(f"   Calls: {len(options_df[options_df['cp'] == 'call'])}")
        print(f"   Puts: {len(options_df[options_df['cp'] == 'put'])}")
        
        # Spread analysis
        options_df['rel_spread'] = (options_df['ask'] - options_df['bid']) / options_df['mid']
        print(f"   Average relative spread: {options_df['rel_spread'].mean():.3f}")
        print(f"   Max relative spread: {options_df['rel_spread'].max():.3f}")
        
        # Strike range
        print(f"   Strike range: {options_df['strike'].min():.0f} - {options_df['strike'].max():.0f}")
        
        # IV analysis
        if 'iv' in options_df.columns:
            print(f"   IV range: {options_df['iv'].min():.3f} - {options_df['iv'].max():.3f}")
            print(f"   Average IV: {options_df['iv'].mean():.3f}")
        
        # Greeks analysis
        greeks = ['delta', 'gamma', 'theta', 'vega']
        for greek in greeks:
            if greek in options_df.columns:
                print(f"   {greek.capitalize()} range: {options_df[greek].min():.4f} - {options_df[greek].max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Options quality analysis error: {e}")
        return False


def main():
    """Main test function."""
    print("Sample FTSE 100 Data Provider Test")
    print("=" * 50)
    print("This test generates realistic sample data for development and testing")
    print("when real data sources are not available.")
    print()
    
    # Run tests
    tests = [
        test_connection,
        test_index_snapshot,
        test_historical_data,
        test_options_data,
        test_data_quality,
        test_options_quality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All tests passed! Sample data provider is working correctly.")
        print("\nYou can now use this for development and testing while you")
        print("set up real data sources for production.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
