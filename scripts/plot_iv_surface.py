#!/usr/bin/env python3
"""
Plot an implied volatility surface from saved options CSVs in data/options/raw/.

- X axis: Expiry (as date)
- Y axis: Strike
- Z axis: Implied Volatility (%)

Examples:
  python scripts/plot_iv_surface.py --ticker SPY --kind calls --num-expiries 8
  python scripts/plot_iv_surface.py --ticker QQQ --kind puts --num-expiries 6
"""

import argparse
import glob
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "options", "raw")


def list_files_for(ticker: str, kind: str) -> List[str]:
    pattern = os.path.join(RAW_DIR, f"{ticker}_*_*_{kind}.csv")
    files = glob.glob(pattern)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def parse_expiry_from_filename(path: str) -> str:
    base = os.path.basename(path).replace(".csv", "")
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""


def collect_points(ticker: str, kind: str, num_expiries: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime]]:
    files = list_files_for(ticker, kind)
    if not files:
        raise FileNotFoundError(f"No {kind} files found for {ticker} in {RAW_DIR}")

    points = []  # (expiry_dt, strike, iv_pct)
    expiries_used = []

    for path in files:
        expiry_str = parse_expiry_from_filename(path)
        if not expiry_str or expiry_str in expiries_used:
            continue
        try:
            exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        except Exception:
            continue
        df = pd.read_csv(path)
        if "impliedVolatility" not in df.columns or "strike" not in df.columns:
            continue
        iv = df["impliedVolatility"].astype(float) * 100.0
        strike = df["strike"].astype(float)
        # Filter obvious outliers/invalids
        mask = np.isfinite(iv) & np.isfinite(strike) & (iv > 0) & (iv < 500)
        iv = iv[mask]
        strike = strike[mask]
        # Append points for this expiry
        for k, v in zip(strike.values, iv.values):
            points.append((exp_dt, k, v))
        expiries_used.append(expiry_str)
        if len(expiries_used) >= num_expiries:
            break

    if not points:
        raise RuntimeError("No valid points collected for surface plot")

    # Convert to arrays
    exp_dates = [p[0] for p in points]
    strikes = np.array([p[1] for p in points], dtype=float)
    ivs = np.array([p[2] for p in points], dtype=float)

    # Convert dates to matplotlib date numbers
    exp_nums = mdates.date2num(exp_dates)
    return np.array(exp_nums, dtype=float), strikes, ivs, sorted(set(exp_dates))


def plot_surface(ticker: str, kind: str, num_expiries: int):
    X, Y, Z, unique_exps = collect_points(ticker, kind, num_expiries)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Use triangulation-based surface on scattered data
    surf = ax.plot_trisurf(X, Y, Z, cmap="viridis", linewidth=0.2, antialiased=True, alpha=0.9)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="IV (%)")

    # Format X axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlabel('Expiry')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title(f"IV Surface: {ticker} ({kind})")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot IV surface from saved options CSVs")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., SPY")
    parser.add_argument("--kind", choices=["calls", "puts"], default="calls")
    parser.add_argument("--num-expiries", type=int, default=8, help="Number of expiries to include")
    args = parser.parse_args()

    plot_surface(args.ticker, args.kind, args.num_expiries)


if __name__ == "__main__":
    main()
