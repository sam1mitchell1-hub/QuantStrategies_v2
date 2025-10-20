#!/usr/bin/env python3
"""
Hybrid script to fetch historical data using yfinance (reliable) 
and optionally integrate with Saxo Bank API for live data.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Major liquid equities for volatility modeling
MAJOR_EQUITIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
    "JPM", "BAC", "WMT", "JNJ", "PG", "UNH", "HD", "V", "MA", "DIS",
    "ADBE", "CRM", "PYPL", "INTC", "AMD", "ORCL", "CSCO", "IBM",
    "SPY", "QQQ", "IWM", "VIX"  # ETFs and volatility index
]


def fetch_yfinance_data(symbols: list, start_date: str, end_date: str, 
                       interval: str = "1d", output_dir: str = "data/raw"):
    """
    Fetch historical data using yfinance (reliable fallback).
    
    Args:
        symbols: List of stock symbols to fetch
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, 5m, 1m)
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Fetching data for {symbol}...")
        
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data is not None and not data.empty:
                # Normalize columns to lowercase
                data.columns = data.columns.str.lower()
                
                # Ensure we have the required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in data.columns for col in required_cols):
                    # Save to CSV
                    filename = f"{symbol}_{start_date}_to_{end_date}_{interval}.csv"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Reset index to include timestamp as a column
                    data_to_save = data.reset_index()
                    data_to_save.to_csv(filepath, index=False)
                    
                    logger.info(f"✅ {symbol}: {len(data)} data points saved to {filename}")
                    
                    # Print summary
                    latest_close = data['close'].iloc[-1] if 'close' in data.columns else "N/A"
                    date_range = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                    logger.info(f"   Date range: {date_range}, Latest close: ${latest_close}")
                    
                    successful += 1
                else:
                    logger.warning(f"❌ {symbol}: Missing required columns")
                    failed += 1
            else:
                logger.warning(f"❌ No data received for {symbol}")
                failed += 1
                
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\nData fetch completed!")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Data saved to: {output_dir}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Fetch historical equity data using yfinance")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (default: major equities)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--interval", type=str, default="1d", choices=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], 
                       help="Data interval (default: 1d)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--major-only", action="store_true", help="Use only major liquid equities")
    
    args = parser.parse_args()
    
    # Determine symbols to fetch
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.major_only:
        symbols = MAJOR_EQUITIES
    else:
        # Default to a subset of major equities for testing
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"]
    
    # Set end date to today if not provided
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    
    print("Historical Data Fetcher (yfinance)")
    print("=" * 50)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Date range: {args.start} to {end_date}")
    print(f"Interval: {args.interval}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Fetch data
    fetch_yfinance_data(symbols, args.start, end_date, args.interval, args.output_dir)


if __name__ == "__main__":
    main()
