#!/usr/bin/env python3
"""
Exploratory tool for options CSVs saved under data/options/raw.
- Lists latest files per ticker/expiry
- Loads a file, shows columns, head, basic stats
- Computes mid price, moneyness, and saves a summarized CSV

Usage examples:
  python scripts/explore_options.py --ticker SPY --kind calls --limit 1
  python scripts/explore_options.py --ticker QQQ --kind puts --limit 1 --save-summary
"""

import argparse
import glob
import os
import sys
from datetime import datetime
import pandas as pd

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join("data", "options", "raw")
SUMMARY_DIR = os.path.join("data", "options", "processed")


def list_files(ticker: str, kind: str) -> list:
    pattern = os.path.join(RAW_DIR, f"{ticker}_*_*_{{kind}}.csv".format(kind=kind))
    files = glob.glob(pattern)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def compute_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # mid price
    if {"bid", "ask"}.issubset(df.columns):
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
    else:
        df["mid"] = df.get("lastPrice", pd.Series([None] * len(df)))
    # yfinance IV column exists and is decimal (e.g., 0.25)
    if "impliedVolatility" in df.columns:
        df["iv_yf"] = df["impliedVolatility"]
    # moneyness proxy requires underlying; not in CSV. Leave placeholder.
    # Users can merge spot later; we include strike and expiry.
    return df


def main():
    parser = argparse.ArgumentParser(description="Explore saved options CSVs")
    parser.add_argument("--ticker", required=True, help="Ticker, e.g., SPY or QQQ")
    parser.add_argument("--kind", choices=["calls", "puts"], default="calls",
                        help="Which side to inspect (default: calls)")
    parser.add_argument("--limit", type=int, default=1, help="How many latest files to load")
    parser.add_argument("--save-summary", action="store_true", help="Save summarized CSV with mid and iv")
    args = parser.parse_args()

    files = list_files(args.ticker, args.kind)
    if not files:
        print(f"No {args.kind} files found for {args.ticker} in {RAW_DIR}")
        sys.exit(1)

    files = files[: args.limit]
    print(f"Found {len(files)} file(s). Showing the most recent: {os.path.basename(files[0])}")

    df = pd.read_csv(files[0])
    print("\nColumns:\n", list(df.columns))
    print("\nHead:\n", df.head(10).to_string(index=False))
    print("\nBasic stats (strike, bid, ask, lastPrice, volume, openInterest):")
    subset_cols = [c for c in ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"] if c in df.columns]
    if subset_cols:
        print(df[subset_cols].describe(include='all').to_string())

    df2 = compute_fields(df)

    if args.save_summary:
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(files[0]).replace(".csv", "")
        out_path = os.path.join(SUMMARY_DIR, f"summary_{base}.csv")
        keep_cols = [c for c in [
            "ticker", "expiry", "contractSymbol", "strike", "type", "lastTradeDate",
            "bid", "ask", "lastPrice", "mid", "impliedVolatility", "iv_yf",
            "volume", "openInterest", "currency"
        ] if c in df2.columns]
        df2[keep_cols].to_csv(out_path, index=False)
        print(f"\nSaved summary to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
