#!/usr/bin/env python3
"""
Plot implied volatility curves from saved options CSVs in data/options/raw/.

Modes:
- smile: plot IV vs strike for a single expiry (calls/puts/both)
- term:  plot IV vs expiry (ATM-ish option per expiry)

Examples:
  python scripts/plot_iv_curve.py --ticker SPY --mode smile --kind both --expiry 2025-09-15
  python scripts/plot_iv_curve.py --ticker QQQ --mode term --num-expiries 6
"""

import argparse
import glob
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join("data", "options", "raw")


def list_files_for(ticker: str, kind: str) -> List[str]:
    pattern = os.path.join(RAW_DIR, f"{ticker}_*_*_{kind}.csv")
    files = glob.glob(pattern)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def parse_expiry_from_filename(path: str) -> str:
    # Filename pattern: TICKER_<expiry>_<timestamp>_<kind>.csv
    base = os.path.basename(path).replace(".csv", "")
    parts = base.split("_")
    # e.g., ["SPY", "2025-09-15", "20250911", "202108", "calls"]
    if len(parts) >= 2:
        return parts[1]
    return ""


def load_expiry_df(ticker: str, expiry: str, kind: str) -> pd.DataFrame:
    files = [p for p in list_files_for(ticker, kind) if parse_expiry_from_filename(p) == expiry]
    if not files:
        raise FileNotFoundError(f"No {kind} CSV found for {ticker} expiry {expiry} in {RAW_DIR}")
    df = pd.read_csv(files[0])
    df["kind"] = kind
    return df


def get_spot(ticker: str) -> float:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return float("nan")


def plot_smile(ticker: str, expiry: str, kind: str):
    to_plot = []
    kinds = [kind] if kind != "both" else ["calls", "puts"]
    for k in kinds:
        df = load_expiry_df(ticker, expiry, k)
        # Compute mid and IV percent
        if {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
        if "impliedVolatility" in df.columns:
            df["iv_pct"] = df["impliedVolatility"] * 100.0
        to_plot.append(df)

    plt.figure(figsize=(10, 6))
    for df in to_plot:
        label = f"{ticker} {expiry} {df['kind'].iloc[0]}"
        plt.scatter(df["strike"], df["iv_pct"], s=16, alpha=0.7, label=label)
    plt.title(f"IV Smile: {ticker} {expiry}")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_term_structure(ticker: str, num_expiries: int, pick_kind: str):
    # Gather most recent expiries from either calls files
    call_files = list_files_for(ticker, "calls")
    puts_files = list_files_for(ticker, "puts")
    files = call_files if pick_kind == "calls" else puts_files
    if not files:
        raise FileNotFoundError(f"No {pick_kind} files found for {ticker}")

    expiries_seen = []
    points: List[Tuple[datetime, float]] = []

    spot = get_spot(ticker)

    for path in files:
        exp = parse_expiry_from_filename(path)
        if exp in expiries_seen:
            continue
        expiries_seen.append(exp)
        df = pd.read_csv(path)
        if "impliedVolatility" not in df.columns or "strike" not in df.columns:
            continue
        # Pick ATM by nearest strike to spot if spot is available; otherwise median strike
        if pd.notna(spot):
            df["kdist"] = (df["strike"] - spot).abs()
            row = df.sort_values("kdist").iloc[0]
        else:
            row = df.iloc[len(df)//2]
        iv = float(row["impliedVolatility"]) * 100.0
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        points.append((exp_dt, iv))
        if len(expiries_seen) >= num_expiries:
            break

    points.sort(key=lambda x: x[0])
    if not points:
        raise RuntimeError("No points to plot for term structure")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o")
    plt.title(f"IV Term Structure ({pick_kind} ATM): {ticker}")
    plt.xlabel("Expiry")
    plt.ylabel("Implied Volatility (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot IV curves from saved options CSVs")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., SPY")
    parser.add_argument("--mode", choices=["smile", "term"], default="smile")
    parser.add_argument("--kind", choices=["calls", "puts", "both"], default="both",
                        help="Which side to use (smile supports both; term uses one side)")
    parser.add_argument("--expiry", help="Expiry date YYYY-MM-DD (required for smile if not inferable)")
    parser.add_argument("--num-expiries", type=int, default=5, help="Number of expiries for term structure")
    args = parser.parse_args()

    if args.mode == "smile":
        expiry = args.expiry
        if not expiry:
            # Try infer the most recent expiry from calls files
            files = list_files_for(args.ticker, "calls")
            if not files:
                raise SystemExit("No files found to infer expiry. Provide --expiry.")
            expiry = parse_expiry_from_filename(files[0])
        plot_smile(args.ticker, expiry, args.kind)
    else:
        kind = args.kind if args.kind != "both" else "calls"
        plot_term_structure(args.ticker, args.num_expiries, kind)


if __name__ == "__main__":
    main()
