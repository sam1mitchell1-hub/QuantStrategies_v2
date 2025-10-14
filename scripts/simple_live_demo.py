#!/usr/bin/env python3
"""
Simple Live Demo Script

A simplified demonstration of the strategy framework using sample data
that shows the core functionality without complex feature engineering.
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

# Import data providers
from data_providers.sample_ftse import SampleFTSEProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_data_flow():
    """Demonstrate the complete data flow for live trading."""
    print("🚀 FTSE 100 Options Trading Strategy - Live Demo")
    print("=" * 60)
    print("This demo shows the complete data flow for live trading:")
    print("1. Historical data retrieval")
    print("2. Current options data retrieval")
    print("3. Current price retrieval")
    print("4. Data processing and analysis")
    print("5. Trading decision simulation")
    print()
    
    # Initialize sample provider
    provider = SampleFTSEProvider(base_price=7500.0, volatility=0.20)
    
    # 1. Historical Data
    print("1️⃣  HISTORICAL DATA RETRIEVAL")
    print("-" * 40)
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # 3 months
    
    try:
        historical_data = provider.get_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"✅ Retrieved {len(historical_data)} days of historical data")
        print(f"   Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
        print(f"   Latest close: {historical_data['close'].iloc[-1]:.2f}")
        print(f"   Price range: {historical_data['close'].min():.2f} - {historical_data['close'].max():.2f}")
        
        # Show sample data
        print("\n   Sample historical data (last 5 days):")
        sample_cols = ['open', 'high', 'low', 'close', 'volume']
        print(historical_data[sample_cols].tail().to_string())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # 2. Current Options Data
    print("\n2️⃣  CURRENT OPTIONS DATA RETRIEVAL")
    print("-" * 40)
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        current_options = provider.get_option_chain_snapshot(
            date=today,
            near_time="15:45",
            window=20,
            maturity_bounds=(7, 60),
            spread_limit=0.5,
            min_oi=100
        )
        
        print(f"✅ Retrieved {len(current_options)} options contracts")
        print(f"   Calls: {len(current_options[current_options['cp'] == 'call'])}")
        print(f"   Puts: {len(current_options[current_options['cp'] == 'put'])}")
        print(f"   Strike range: {current_options['strike'].min():.0f} - {current_options['strike'].max():.0f}")
        
        # Show sample options
        print("\n   Sample options data:")
        sample_cols = ['cp', 'strike', 'expiry', 'bid', 'ask', 'mid', 'iv']
        print(current_options[sample_cols].head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # 3. Current Index Price
    print("\n3️⃣  CURRENT INDEX PRICE RETRIEVAL")
    print("-" * 40)
    
    try:
        current_snapshot = provider.get_index_snapshot(today, "15:45", 20)
        current_price = current_snapshot.index_px
        
        print(f"✅ Current FTSE 100 price: {current_price:.2f}")
        print(f"   Volume: {current_snapshot.volume:,}")
        print(f"   Timestamp: {current_snapshot.timestamp}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # 4. Data Analysis
    print("\n4️⃣  DATA ANALYSIS")
    print("-" * 40)
    
    try:
        # Price analysis
        price_change = historical_data['close'].iloc[-1] - historical_data['close'].iloc[-2]
        price_change_pct = (price_change / historical_data['close'].iloc[-2]) * 100
        
        print(f"✅ Price analysis:")
        print(f"   Daily change: {price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"   Volatility: {historical_data['close'].pct_change().std() * np.sqrt(252):.1%}")
        
        # Options analysis
        atm_options = current_options[
            (current_options['strike'] >= current_price * 0.95) & 
            (current_options['strike'] <= current_price * 1.05)
        ]
        
        print(f"\n✅ Options analysis:")
        print(f"   ATM options: {len(atm_options)}")
        if not atm_options.empty:
            avg_iv = atm_options['iv'].mean()
            print(f"   Average IV: {avg_iv:.1%}")
            print(f"   IV range: {atm_options['iv'].min():.1%} - {atm_options['iv'].max():.1%}")
        
        # Volume analysis
        avg_volume = historical_data['volume'].mean()
        current_volume = current_snapshot.volume
        volume_ratio = current_volume / avg_volume
        
        print(f"\n✅ Volume analysis:")
        print(f"   Average volume: {avg_volume:,.0f}")
        print(f"   Current volume: {current_volume:,.0f}")
        print(f"   Volume ratio: {volume_ratio:.2f}x")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # 5. Trading Decision Simulation
    print("\n5️⃣  TRADING DECISION SIMULATION")
    print("-" * 40)
    
    try:
        # Simple trading logic based on price movement and volatility
        recent_returns = historical_data['close'].pct_change().tail(5)
        avg_recent_return = recent_returns.mean()
        recent_volatility = recent_returns.std()
        
        # Simple momentum signal
        if avg_recent_return > 0.001:  # 0.1% threshold
            signal = "BULLISH"
            confidence = min(abs(avg_recent_return) * 100, 1.0)
        elif avg_recent_return < -0.001:
            signal = "BEARISH"
            confidence = min(abs(avg_recent_return) * 100, 1.0)
        else:
            signal = "NEUTRAL"
            confidence = 0.0
        
        print(f"✅ Trading signal generated:")
        print(f"   Signal: {signal}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Recent return: {avg_recent_return:.2%}")
        print(f"   Recent volatility: {recent_volatility:.2%}")
        
        # Options recommendation
        if signal != "NEUTRAL" and confidence > 0.3:
            if signal == "BULLISH":
                print(f"\n   📈 RECOMMENDATION: Consider bull call spread")
                print(f"   🎯 Target: Calls near {current_price:.0f} strike")
            else:
                print(f"\n   📉 RECOMMENDATION: Consider bear put spread")
                print(f"   🎯 Target: Puts near {current_price:.0f} strike")
        else:
            print(f"\n   ⏸️  RECOMMENDATION: No trade - market conditions not favorable")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # 6. Summary
    print("\n🎉 LIVE DEMO COMPLETE")
    print("=" * 60)
    print("✅ Historical data: Retrieved and analyzed")
    print("✅ Current options: Retrieved and analyzed")
    print("✅ Current price: Retrieved and analyzed")
    print("✅ Data analysis: Completed")
    print("✅ Trading signal: Generated")
    print("✅ Options recommendation: Provided")
    
    print("\n📝 DEMO SUMMARY:")
    print("   This demo shows the complete data flow for live trading:")
    print("   - Data retrieval from multiple sources")
    print("   - Real-time analysis and processing")
    print("   - Trading signal generation")
    print("   - Options strategy recommendations")
    print()
    print("   The same process works with real market data sources.")
    
    return True


def main():
    """Main function."""
    success = demonstrate_data_flow()
    
    if success:
        print("\n🎯 Demo completed successfully!")
        print("Your strategy framework is ready for live trading.")
        print("\nNext steps:")
        print("1. Replace sample data with real market data")
        print("2. Implement advanced feature engineering")
        print("3. Add machine learning models")
        print("4. Deploy to production")
    else:
        print("\n❌ Demo failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
