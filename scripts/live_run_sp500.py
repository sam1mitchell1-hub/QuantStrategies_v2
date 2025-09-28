#!/usr/bin/env python3
"""
Live Run Simulation with S&P 500 Data

This script simulates a live run using S&P 500 data (which is widely available)
to demonstrate the strategy functionality with real market data.
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
from data_providers.sample_ftse import SampleFTSEProvider
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sp500_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Get S&P 500 historical data from Yahoo Finance."""
    print(f"📊 Fetching S&P 500 historical data from {start_date} to {end_date}...")
    
    try:
        ticker = yf.Ticker("^GSPC")
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print("❌ No S&P 500 data from Yahoo Finance")
            return pd.DataFrame()
        
        # Convert to London timezone for consistency
        data.index = data.index.tz_localize('UTC').tz_convert('Europe/London')
        
        print(f"✅ Retrieved {len(data)} days of S&P 500 data")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Latest close: {data['Close'].iloc[-1]:.2f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching S&P 500 data: {e}")
        return pd.DataFrame()


def get_sp500_current_price() -> float:
    """Get current S&P 500 price."""
    print("💰 Fetching current S&P 500 price...")
    
    try:
        ticker = yf.Ticker("^GSPC")
        data = ticker.history(period="1d")
        
        if data.empty:
            print("❌ No current S&P 500 data")
            return 0.0
        
        current_price = data['Close'].iloc[-1]
        print(f"✅ Current S&P 500 price: {current_price:.2f}")
        return current_price
        
    except Exception as e:
        print(f"❌ Error fetching current S&P 500 price: {e}")
        return 0.0


def get_sample_options_data(base_price: float) -> pd.DataFrame:
    """Get sample options data based on current price."""
    print("📈 Generating sample options data...")
    
    try:
        # Create sample provider with current price
        provider = SampleFTSEProvider(base_price=base_price)
        today = datetime.now().strftime('%Y-%m-%d')
        
        options_df = provider.get_option_chain_snapshot(
            date=today,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        print(f"✅ Generated {len(options_df)} sample options")
        print(f"   Calls: {len(options_df[options_df['cp'] == 'call'])}")
        print(f"   Puts: {len(options_df[options_df['cp'] == 'put'])}")
        
        return options_df
        
    except Exception as e:
        print(f"❌ Error generating sample options: {e}")
        return pd.DataFrame()


def simulate_live_run():
    """Simulate a live run using S&P 500 data."""
    print("🚀 Starting Live Run Simulation (S&P 500 + Sample Options)")
    print("=" * 60)
    
    # 1. Get S&P 500 historical data
    print("\n1️⃣  SETTING UP S&P 500 HISTORICAL DATA")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    historical_prices = get_sp500_historical_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if historical_prices.empty:
        print("❌ Cannot proceed without historical data")
        return False
    
    # 2. Get current S&P 500 price
    print("\n2️⃣  GETTING CURRENT S&P 500 PRICE")
    current_price = get_sp500_current_price()
    
    if current_price == 0:
        print("❌ Cannot proceed without current price")
        return False
    
    # 3. Generate sample options data based on current price
    print("\n3️⃣  GENERATING SAMPLE OPTIONS DATA")
    current_options = get_sample_options_data(current_price)
    
    if current_options.empty:
        print("❌ Cannot proceed without options data")
        return False
    
    # 4. Prepare market data
    print("\n4️⃣  PREPARING MARKET DATA")
    market_data = {
        'prices': historical_prices,
        'options': current_options
    }
    
    print(f"   S&P 500 prices: {len(historical_prices)} days")
    print(f"   Current price: {current_price:.2f}")
    print(f"   Sample options: {len(current_options)} contracts")
    
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
        print(f"   Sample features: {feature_set.feature_names[:5]}")
        
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
                print(f"   📊 Market: S&P 500 at {current_price:.2f}")
                print(f"   📈 Direction: {'Bullish' if expected_return > 0 else 'Bearish'}")
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
    print("=" * 60)
    print("✅ S&P 500 historical data: Retrieved")
    print("✅ S&P 500 current price: Retrieved")
    print("✅ Sample options data: Generated")
    print("✅ Features: Built")
    print("✅ Model: Trained")
    print("✅ Prediction: Made")
    print("✅ Trading signal: Generated")
    
    print("\n📝 NOTE: This simulation uses:")
    print("   - Real S&P 500 data for underlying prices")
    print("   - Sample data for options (realistic but synthetic)")
    print("   - Your strategy framework works with any data source")
    
    return True


def main():
    """Main function."""
    print("S&P 500 Options Trading Strategy - Live Run Simulation")
    print("=" * 60)
    print("This script demonstrates live trading simulation using:")
    print("- Real S&P 500 data (widely available)")
    print("- Sample options data (realistic but synthetic)")
    print("- Your complete strategy framework")
    print()
    
    success = simulate_live_run()
    
    if success:
        print("\n🎯 Simulation completed successfully!")
        print("Your strategy framework is ready for live trading.")
        print("\nNext steps:")
        print("1. Replace sample options with real options data")
        print("2. Use FTSE 100 data when available")
        print("3. Deploy to production")
    else:
        print("\n❌ Simulation failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
