#!/usr/bin/env python3
"""
Quick Live Test Script

A simplified script to test the core functionality of the strategy
with real market data where possible.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import logging

# Import data providers
from data_providers.yahoo_ftse import YahooFTSEProvider
from data_providers.sample_ftse import SampleFTSEProvider
from data_providers.ftse_data_manager import FTSEDataManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_yahoo_ftse_data():
    """Test Yahoo Finance FTSE 100 data."""
    print("🔍 Testing Yahoo Finance FTSE 100 Data")
    print("-" * 40)
    
    try:
        provider = YahooFTSEProvider()
        
        # Test historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        print(f"Fetching historical data from {start_date} to {end_date}...")
        historical_data = provider.get_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if historical_data.empty:
            print("❌ No historical data from Yahoo Finance")
            return False
        
        print(f"✅ Historical data: {len(historical_data)} days")
        print(f"   Latest close: {historical_data['close'].iloc[-1]:.2f}")
        print(f"   Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
        
        # Test current snapshot
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"\nFetching current snapshot for {today}...")
        snapshot = provider.get_index_snapshot(today, "15:45", 20)
        
        print(f"✅ Current snapshot: {snapshot.index_px:.2f}")
        print(f"   Volume: {snapshot.volume:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_options_data():
    """Test options data availability."""
    print("\n🔍 Testing Options Data Availability")
    print("-" * 40)
    
    try:
        # Try unified data manager
        manager = FTSEDataManager()
        working_provider = manager.get_working_provider()
        
        if working_provider:
            print(f"✅ Working provider: {working_provider}")
            
            today = datetime.now().strftime('%Y-%m-%d')
            options_df = manager.get_option_chain_snapshot(
                date=today,
                near_time="15:45",
                window=20,
                maturity_bounds=(7, 60),
                spread_limit=0.5,
                min_oi=100
            )
            
            if not options_df.empty:
                print(f"✅ Options data: {len(options_df)} contracts")
                print(f"   Calls: {len(options_df[options_df['cp'] == 'call'])}")
                print(f"   Puts: {len(options_df[options_df['cp'] == 'put'])}")
                return True
            else:
                print("⚠️  No options data from working provider")
        else:
            print("⚠️  No working providers found")
        
        # Fallback to sample data
        print("Using sample data for options...")
        sample_provider = SampleFTSEProvider()
        today = datetime.now().strftime('%Y-%m-%d')
        options_df = sample_provider.get_option_chain_snapshot(
            date=today,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        print(f"✅ Sample options data: {len(options_df)} contracts")
        print(f"   Calls: {len(options_df[options_df['cp'] == 'call'])}")
        print(f"   Puts: {len(options_df[options_df['cp'] == 'put'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_strategy_integration():
    """Test strategy integration with real data."""
    print("\n🔍 Testing Strategy Integration")
    print("-" * 40)
    
    try:
        # Get historical data
        provider = YahooFTSEProvider()
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        
        historical_data = provider.get_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if historical_data.empty:
            print("❌ No historical data for strategy test")
            return False
        
        # Get current options data
        sample_provider = SampleFTSEProvider()
        today = datetime.now().strftime('%Y-%m-%d')
        current_options = sample_provider.get_option_chain_snapshot(
            date=today,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        # Prepare market data
        market_data = {
            'prices': historical_data,
            'options': current_options
        }
        
        print(f"✅ Market data prepared:")
        print(f"   Historical prices: {len(historical_data)} days")
        print(f"   Current options: {len(current_options)} contracts")
        
        # Test feature building
        from strategy import StrategyConfig, FeatureBuilder
        
        config = StrategyConfig()
        feature_builder = FeatureBuilder(config)
        
        print("Building features...")
        feature_set = feature_builder.build_features(**market_data)
        
        print(f"✅ Features built: {len(feature_set.feature_names)} features")
        print(f"   Sample features: {feature_set.feature_names[:5]}")
        
        # Test model training
        from strategy import GBTForecaster
        
        forecaster = GBTForecaster(config)
        print("Training model...")
        model_results = forecaster.train(feature_set.features, feature_set.target_direction)
        
        print(f"✅ Model trained: {model_results.metrics.get('accuracy', 'N/A'):.3f} accuracy")
        
        # Test prediction
        latest_features = feature_set.features.iloc[-1:].copy()
        forecast_results = forecaster.predict(latest_features)
        
        print(f"✅ Prediction made:")
        print(f"   Prediction: {forecast_results.predictions[0]}")
        if forecast_results.probabilities is not None:
            print(f"   Probability: {forecast_results.probabilities[0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("FTSE 100 Strategy - Quick Live Test")
    print("=" * 50)
    print("Testing core functionality with real market data")
    print()
    
    # Run tests
    tests = [
        ("Yahoo Finance Data", test_yahoo_ftse_data),
        ("Options Data", test_options_data),
        ("Strategy Integration", test_strategy_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your strategy is ready for live trading.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
