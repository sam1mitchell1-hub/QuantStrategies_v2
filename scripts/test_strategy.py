#!/usr/bin/env python3
"""
Test script for the Bollinger Bands + RSI strategy.
Demonstrates how to use the strategy with sample data.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_strategies.strategies import BollingerRSIStrategy
from quant_strategies.data import DataManager

def main():
    print("Bollinger Bands + RSI Strategy Test")
    print("====================================")
    
    # Initialize the strategy
    strategy = BollingerRSIStrategy()
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.parameters}")
    print()
    
    # Load sample data (you can replace this with your own data)
    print("Loading sample data...")
    
    # Option 1: Use DataManager to fetch fresh data
    data_manager = DataManager()
    tickers = ['AAPL']  # Test with one ticker first
    
    try:
        # Fetch 5 years of data with end date as yesterday
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=5*365)  # 5 years ago
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching data from {start_str} to {end_str}")
        
        # Use the save_daily_prices method with specific date range
        file_path = data_manager.client.save_daily_prices(tickers[0], start_str, end_str, 
                                                        output_dir=data_manager.raw_data_dir)
        
        if file_path:
            # Load the data
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} data points for {tickers[0]}")
        else:
            print("Failed to fetch data, using sample data instead")
            df = create_sample_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using sample data instead")
        df = create_sample_data()
    
    # Validate data
    if not strategy.validate_data(df):
        print("Data validation failed!")
        return
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Run the strategy
    print("Running strategy backtest...")
    results = strategy.backtest(df)
    
    # Display results
    print("\nBacktest Results:")
    print("=================")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Get detailed data with indicators and signals
    print("\nCalculating indicators and signals...")
    df_with_indicators = strategy.calculate_indicators(df)
    df_with_signals = strategy.generate_signals(df_with_indicators)
    
    # Show some signal examples
    signals = df_with_signals[df_with_signals['signal'] != 'HOLD']
    print(f"\nFound {len(signals)} trading signals:")
    
    if len(signals) > 0:
        print("\nSample signals:")
        for idx, row in signals.head(10).iterrows():
            print(f"  {row['date']}: {row['signal']} at ${row['close']:.2f} "
                  f"(RSI: {row['rsi']:.1f}, Strength: {row['signal_strength']:.3f})")
    
    print(f"\nStrategy test completed!")

def create_sample_data():
    """Create sample OHLCV data for testing."""
    import numpy as np
    
    # Create date range
    dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')
    
    # Create sample price data (random walk)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = {
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main() 