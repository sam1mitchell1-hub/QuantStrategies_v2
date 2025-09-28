#!/usr/bin/env python3
"""
FTSE 100 Strategy Execution Script

This script demonstrates how to run the options trading strategy
using real FTSE 100 data from Intrinio.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import logging

from data_providers.intrinio_ftse import IntrinioFTSEProvider
from strategy import StrategyConfig, FeatureBuilder, GBTForecaster, Backtester
from data_providers.config import DataProviderConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_ftse_price_data(index_snapshots):
    """
    Create price data DataFrame from index snapshots.
    
    Args:
        index_snapshots: List of IndexSnapshot objects
        
    Returns:
        pd.DataFrame: OHLCV price data
    """
    data = []
    for snapshot in index_snapshots:
        data.append({
            'timestamp': snapshot.timestamp,
            'open': snapshot.index_px,  # Simplified: using same price for OHLC
            'high': snapshot.index_px,
            'low': snapshot.index_px,
            'close': snapshot.index_px,
            'volume': snapshot.volume or 0
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def create_ftse_options_data(options_snapshots):
    """
    Create options data DataFrame from options snapshots.
    
    Args:
        options_snapshots: List of DataFrames with options data
        
    Returns:
        pd.DataFrame: Options chain data
    """
    if not options_snapshots:
        return pd.DataFrame()
    
    # Combine all options data
    combined_df = pd.concat(options_snapshots, ignore_index=True)
    
    # Create IV surface features
    options_features = pd.DataFrame(index=combined_df['timestamp'].unique())
    
    # ATM IV
    atm_options = combined_df[combined_df['strike'].between(combined_df['strike'].quantile(0.45), 
                                                           combined_df['strike'].quantile(0.55))]
    if not atm_options.empty:
        options_features['iv_atm'] = atm_options.groupby('timestamp')['iv'].mean()
    
    # IV skew features
    for strike_pct in [0.9, 0.95, 1.0, 1.05, 1.1]:
        strike_level = combined_df['strike'].quantile(strike_pct)
        skew_options = combined_df[combined_df['strike'].between(strike_level * 0.98, strike_level * 1.02)]
        if not skew_options.empty:
            options_features[f'iv_skew_{strike_pct:.0%}'] = skew_options.groupby('timestamp')['iv'].mean()
    
    # Volume and OI features
    options_features['total_oi'] = combined_df.groupby('timestamp')['oi'].sum()
    options_features['avg_bid_ask_spread'] = combined_df.groupby('timestamp').apply(
        lambda x: ((x['ask'] - x['bid']) / x['mid']).mean()
    )
    
    return options_features


def run_ftse_strategy(start_date: str, end_date: str, config_path: str = None):
    """
    Run the options trading strategy with FTSE 100 data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config_path: Path to strategy configuration file
    """
    logger.info(f"Running FTSE 100 strategy from {start_date} to {end_date}")
    
    # Load configuration
    if config_path:
        config = StrategyConfig.from_yaml(config_path)
    else:
        config = StrategyConfig()
    
    # Initialize data provider
    try:
        ftse_provider = IntrinioFTSEProvider()
        if not ftse_provider.test_connection():
            logger.error("Failed to connect to Intrinio API")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize FTSE provider: {e}")
        return False
    
    # Generate trading dates
    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    logger.info(f"Processing {len(trading_dates)} trading days")
    
    # Collect data
    index_snapshots = []
    options_snapshots = []
    
    for date in trading_dates:
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"Fetching data for {date_str}")
        
        try:
            # Get index snapshot
            index_snapshot = ftse_provider.get_index_snapshot(
                date=date_str,
                near_time="15:45",
                window=20
            )
            index_snapshots.append(index_snapshot)
            
            # Get options chain
            options_df = ftse_provider.get_option_chain_snapshot(
                date=date_str,
                near_time="15:45",
                window=20,
                maturity_bounds=(7, 60),
                spread_limit=0.5,
                min_oi=100
            )
            if not options_df.empty:
                options_snapshots.append(options_df)
            
        except Exception as e:
            logger.warning(f"Failed to fetch data for {date_str}: {e}")
            continue
    
    if not index_snapshots:
        logger.error("No index data collected")
        return False
    
    # Create market data
    price_data = create_ftse_price_data(index_snapshots)
    options_data = create_ftse_options_data(options_snapshots)
    
    logger.info(f"Created price data: {len(price_data)} records")
    logger.info(f"Created options data: {len(options_data)} records")
    
    # Initialize strategy components
    feature_builder = FeatureBuilder(config)
    forecaster = GBTForecaster(config)
    backtester = Backtester(config)
    
    # Build features
    logger.info("Building features...")
    market_data = {
        'prices': price_data,
        'options': options_data
    }
    
    feature_set = feature_builder.build_features(**market_data)
    logger.info(f"Built {len(feature_set.feature_names)} features")
    
    # Train model
    logger.info("Training model...")
    model_results = forecaster.train(feature_set.features, feature_set.target_direction)
    logger.info(f"Model training completed. Accuracy: {model_results.metrics['accuracy']:.3f}")
    
    # Run backtest
    logger.info("Running backtest...")
    backtest_results = backtester.run_backtest(
        feature_data=feature_set.features,
        market_data=market_data,
        model=forecaster.model,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date)
    )
    
    # Print results
    print("\n=== FTSE 100 STRATEGY RESULTS ===")
    print(f"Total Trades: {backtest_results.performance_metrics['total_trades']}")
    print(f"Hit Rate: {backtest_results.performance_metrics['hit_rate']:.2%}")
    print(f"Total P&L: ${backtest_results.performance_metrics['total_pnl']:.2f}")
    print(f"Total Return: {backtest_results.performance_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results.performance_metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {backtest_results.performance_metrics['max_drawdown']:.2%}")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FTSE 100 Options Strategy')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Strategy configuration file')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_date = pd.to_datetime(args.start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(args.end_date).strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        return False
    
    # Run strategy
    success = run_ftse_strategy(start_date, end_date, args.config)
    
    if success:
        logger.info("Strategy execution completed successfully!")
    else:
        logger.error("Strategy execution failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
