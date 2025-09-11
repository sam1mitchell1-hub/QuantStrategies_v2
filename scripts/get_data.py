#!/usr/bin/env python3
"""
Data collection script for QuantStrategies project.
Fetches daily price data for specified tickers and saves to data directory.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_strategies.data import DataManager

def main():
    parser = argparse.ArgumentParser(description='Fetch stock data for trading strategies')
    parser.add_argument('--tickers', nargs='+', 
                       default=['TSLA', 'AAPL', 'PLTR'],
                       help='List of ticker symbols (default: TSLA AAPL PLTR)')
    parser.add_argument('--start-date', 
                       help='Start date in YYYY-MM-DD format (default: 5 years ago)')
    parser.add_argument('--data-dir', default='data',
                       help='Directory to store data (default: data)')
    
    args = parser.parse_args()
    
    # Calculate start date if not provided
    if not args.start_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        args.start_date = start_date.strftime('%Y-%m-%d')
    
    print(f"QuantStrategies Data Collection")
    print(f"================================")
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Date range: {args.start_date} to today")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Initialize data manager
    data_manager = DataManager(data_dir=args.data_dir)
    
    # Fetch data for all tickers
    results = data_manager.get_multiple_tickers(args.tickers, args.start_date)
    
    # Summary
    successful = [t for t, path in results.items() if path is not None]
    failed = [t for t, path in results.items() if path is None]
    
    print(f"\nData Collection Complete!")
    print(f"=========================")
    print(f"Successful: {len(successful)} tickers")
    if successful:
        print(f"✓ {', '.join(successful)}")
    
    if failed:
        print(f"Failed: {len(failed)} tickers")
        print(f"✗ {', '.join(failed)}")
    
    print(f"\nData files saved in: {args.data_dir}/raw/")
    print(f"Metadata saved in: {args.data_dir}/metadata/")

if __name__ == "__main__":
    main() 