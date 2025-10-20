#!/usr/bin/env python3
"""
CLI script to run equity OMS backtests.
"""
import argparse
from datetime import datetime, timedelta
from quant_strategies.backtest.equity_oms_backtester import EquityOMSBacktester, ExecConfig
from quant_strategies.strategies import BollingerRSIStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Run equity OMS backtest")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers (e.g., AAPL,MSFT)")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash")
    parser.add_argument("--lookback", type=int, default=100, help="Lookback bars for indicators")
    parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1h"], help="Data interval")
    parser.add_argument("--equity-csv", type=str, default="output/equity_curve.csv", help="Equity curve CSV path")
    parser.add_argument("--trades-csv", type=str, default="output/trades.csv", help="Trades CSV path")
    parser.add_argument("--equity-png", type=str, default="output/equity_curve.png", help="Equity curve PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    exec_cfg = ExecConfig(model="next_open")  # as requested

    backtester = EquityOMSBacktester(
        initial_cash=args.initial_cash,
        exec_config=exec_cfg,
        ledger_path="data/equity_oms_backtest.db"
    )

    strategy = BollingerRSIStrategy()  # reuse existing simple strategy

    results = backtester.run(
        tickers=tickers,
        start=args.start,
        end=args.end,
        strategy=strategy,
        lookback_bars=args.lookback,
        interval=args.interval,
        save_equity_csv=args.equity_csv,
        save_trades_csv=args.trades_csv,
        save_equity_png=args.equity_png
    )

    eq = results["equity_curve"]
    print("\nBacktest complete.")
    if not eq.empty:
        start_val = eq["total_value"].iloc[0]
        end_val = eq["total_value"].iloc[-1]
        ret = (end_val - start_val) / start_val if start_val else 0.0
        print(f"Start value: ${start_val:,.2f}")
        print(f"End value:   ${end_val:,.2f}")
        print(f"Return:      {ret*100:.2f}%")
        print(f"Equity CSV:  {args.equity_csv}")
        print(f"Equity PNG:  {args.equity_png}")
        print(f"Trades CSV:  {args.trades_csv}")
        print(f"Ledger DB:   data/equity_oms_backtest.db")
    else:
        print("No equity data produced (likely no data or signals).")


if __name__ == "__main__":
    main()
