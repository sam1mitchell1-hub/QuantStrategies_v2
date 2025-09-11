#!/usr/bin/env python3
"""
Fetch options chains for SPY and QQQ (or custom tickers) and save CSVs.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_strategies.data import OptionsClient

def main():
    parser = argparse.ArgumentParser(description="Fetch options chains via yfinance")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"],
                        help="Tickers to fetch (default: SPY QQQ)")
    parser.add_argument("--num-expiries", type=int, default=3,
                        help="Number of near expiries to fetch (default: 3)")
    args = parser.parse_args()

    client = OptionsClient()

    for ticker in args.tickers:
        print(f"\nFetching {ticker} options (first {args.num_expiries} expiries)...")
        saved = client.fetch_near_expiries(ticker, num_expiries=args.num_expiries)
        total_calls = len(saved.get("calls", []))
        total_puts = len(saved.get("puts", []))
        print(f"Saved {total_calls} call CSVs and {total_puts} put CSVs for {ticker}")

    print("\nDone. Files saved under data/options/raw/")

if __name__ == "__main__":
    main()
