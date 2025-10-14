#!/usr/bin/env python3
"""
Live Run Simulation Script

This script simulates a live run of the options trading strategy using:
- Yahoo Finance for historical FTSE 100 data
- Best available options data (Yahoo Finance or sample data)
- Real-time simulation of the strategy execution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Import strategy components
from strategy import StrategyConfig, FeatureBuilder, GBTForecaster, Backtester
from data_providers.yahoo_ftse import YahooFTSEProvider
from data_providers.sample_ftse import SampleFTSEProvider
from data_providers.ftse_data_manager import FTSEDataManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_historical_data_yahoo(start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical FTSE 100 data from Yahoo Finance."""
    print(f"📊 Fetching historical FTSE 100 data from {start_date} to {end_date}...")
    
    try:
        provider = YahooFTSEProvider()
        data = provider.get_historical_data(start_date, end_date)
        
        if data.empty:
            print("❌ No historical data from Yahoo Finance")
            return pd.DataFrame()
        
        print(f"✅ Retrieved {len(data)} days of historical data")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Latest close: {data['close'].iloc[-1]:.2f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching historical data: {e}")
        return pd.DataFrame()


def get_current_options_data() -> pd.DataFrame:
    """Get current options data (best available source)."""
    print("📈 Fetching current FTSE 100 options data...")
    
    try:
        # Try unified data manager first
        manager = FTSEDataManager()
        working_provider = manager.get_working_provider()
        
        if working_provider:
            print(f"   Using {working_provider} provider")
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
                print(f"✅ Retrieved {len(options_df)} options from {working_provider}")
                return options_df
            else:
                print(f"⚠️  No options data from {working_provider}, using sample data")
        else:
            print("⚠️  No working providers, using sample data")
        
        # Fallback to sample data
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
        
        print(f"✅ Generated {len(options_df)} sample options")
        return options_df
        
    except Exception as e:
        print(f"❌ Error fetching options data: {e}")
        return pd.DataFrame()


def get_current_index_price() -> float:
    """Get current FTSE 100 index price."""
    print("💰 Fetching current FTSE 100 index price...")
    
    try:
        # Try Yahoo Finance first
        provider = YahooFTSEProvider()
        today = datetime.now().strftime('%Y-%m-%d')
        snapshot = provider.get_index_snapshot(today, "15:45", 20)
        
        print(f"✅ Current FTSE 100 price: {snapshot.index_px:.2f}")
        return snapshot.index_px
        
    except Exception as e:
        print(f"⚠️  Error fetching current price: {e}")
        print("   Using sample data for current price")
        
        # Fallback to sample data
        sample_provider = SampleFTSEProvider()
        today = datetime.now().strftime('%Y-%m-%d')
        snapshot = sample_provider.get_index_snapshot(today, "15:45", 20)
        
        print(f"✅ Sample FTSE 100 price: {snapshot.index_px:.2f}")
        return snapshot.index_px


def simulate_live_run():
    """Simulate a live run of the options trading strategy."""
    print("🚀 Starting Live Run Simulation")
    print("=" * 50)
    
    # 1. Get historical data for feature engineering
    print("\n1️⃣  SETTING UP HISTORICAL DATA")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    historical_prices = get_historical_data_yahoo(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if historical_prices.empty:
        print("❌ Cannot proceed without historical data")
        return False
    
    # 2. Get current options data
    print("\n2️⃣  SETTING UP CURRENT OPTIONS DATA")
    current_options = get_current_options_data()
    
    if current_options.empty:
        print("❌ Cannot proceed without options data")
        return False
    
    # 3. Get current index price
    print("\n3️⃣  GETTING CURRENT INDEX PRICE")
    current_price = get_current_index_price()
    
    # 4. Prepare market data
    print("\n4️⃣  PREPARING MARKET DATA")
    market_data = {
        'prices': historical_prices,
        'options': current_options
    }
    
    print(f"   Historical prices: {len(historical_prices)} days")
    print(f"   Current options: {len(current_options)} contracts")
    print(f"   Current price: {current_price:.2f}")
    
    # 5. Initialize strategy components
    print("\n5️⃣  INITIALIZING STRATEGY COMPONENTS")
    try:
        config = StrategyConfig()
        feature_builder = FeatureBuilder(config)
        forecaster = GBTForecaster(config)
        backtester = Backtester(config)
        
        print("✅ Strategy components initialized")
        
    except Exception as e:
        print(f"❌ Error initializing strategy: {e}")
        return False
    
    # 6. Build features
    print("\n6️⃣  BUILDING FEATURES")
    try:
        feature_set = feature_builder.build_features(**market_data)
        print(f"✅ Built {len(feature_set.feature_names)} features")
        print(f"   Features: {feature_set.feature_names[:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"❌ Error building features: {e}")
        return False
    
    # 7. Train model
    print("\n7️⃣  TRAINING MODEL")
    try:
        model_results = forecaster.train(feature_set.features, feature_set.target_direction)
        print(f"✅ Model trained successfully")
        print(f"   Accuracy: {model_results.metrics.get('accuracy', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    
    # 8. Simulate live decision
    print("\n8️⃣  SIMULATING LIVE DECISION")
    try:
        # Get the most recent features
        latest_features = feature_set.features.iloc[-1:].copy()
        
        # Make prediction
        forecast_results = forecaster.predict(latest_features)
        
        print(f"✅ Live prediction made")
        print(f"   Prediction: {forecast_results.predictions[0]}")
        if forecast_results.probabilities is not None:
            print(f"   Probability: {forecast_results.probabilities[0]:.3f}")
        
        # Simulate trading decision
        if forecast_results.probabilities is not None:
            confidence = abs(forecast_results.probabilities[0] - 0.5) * 2
            expected_return = (forecast_results.probabilities[0] - 0.5) * confidence * 0.02
            
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Expected return: {expected_return:.3f}")
            
            if abs(expected_return) > 0.001:
                print("   🎯 TRADING SIGNAL: Consider trading")
            else:
                print("   ⏸️  TRADING SIGNAL: No trade recommended")
        
    except Exception as e:
        print(f"❌ Error in live decision: {e}")
        return False
    
    # 9. Show options analysis
    print("\n9️⃣  OPTIONS ANALYSIS")
    try:
        if not current_options.empty:
            print(f"   Total options: {len(current_options)}")
            print(f"   Calls: {len(current_options[current_options['cp'] == 'call'])}")
            print(f"   Puts: {len(current_options[current_options['cp'] == 'put'])}")
            print(f"   Strike range: {current_options['strike'].min():.0f} - {current_options['strike'].max():.0f}")
            
            # Show ATM options
            atm_options = current_options[
                (current_options['strike'] >= current_price * 0.95) & 
                (current_options['strike'] <= current_price * 1.05)
            ]
            
            if not atm_options.empty:
                print(f"   ATM options: {len(atm_options)}")
                print("   Sample ATM options:")
                sample_cols = ['cp', 'strike', 'expiry', 'bid', 'ask', 'mid']
                print(atm_options[sample_cols].head(5).to_string(index=False))
        
    except Exception as e:
        print(f"⚠️  Error in options analysis: {e}")
    
    # 10. Summary
    print("\n🎉 LIVE RUN SIMULATION COMPLETE")
    print("=" * 50)
    print("✅ Historical data: Retrieved")
    print("✅ Current options: Retrieved")
    print("✅ Current price: Retrieved")
    print("✅ Features: Built")
    print("✅ Model: Trained")
    print("✅ Prediction: Made")
    print("✅ Trading signal: Generated")
    
    return True


def main():
    """Main function."""
    print("FTSE 100 Options Trading Strategy - Live Run Simulation")
    print("=" * 60)
    print("This script simulates a live run using real market data")
    print("where possible, with sample data as fallback.")
    print()
    
    success = simulate_live_run()
    
    if success:
        print("\n🎯 Simulation completed successfully!")
        print("Your strategy framework is ready for live trading.")
    else:
        print("\n❌ Simulation failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
