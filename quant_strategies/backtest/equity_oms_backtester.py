import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf

from quant_strategies.execution import (
    Portfolio, Ledger, PositionSizer, OrderManagementSystem
)
from quant_strategies.execution.models import SignalType
from quant_strategies.strategies import BollingerRSIStrategy


@dataclass
class ExecConfig:
    commission_per_share: float = 0.005
    min_commission: float = 1.00
    slippage_bps: float = 5.0
    model: str = "next_open"  # execution at next bar open


class BacktestBrokerAgent:
    """
    Backtest broker that fills at the next bar's open with optional slippage and commission.
    """
    def __init__(self, exec_config: ExecConfig):
        self.exec = exec_config

    def execute_order(self, order, current_market_data: Dict) -> List:
        # Use provided next_open from market data
        next_open = current_market_data.get("next_open")
        if next_open is None or next_open <= 0:
            return []

        # Apply side-specific price with slippage
        px = float(next_open)
        if order.side.value == "BUY":
            execution_price = px * (1 + self.exec.slippage_bps / 10000.0)
        else:
            execution_price = px * (1 - self.exec.slippage_bps / 10000.0)

        # Commission
        commission = max(order.quantity * self.exec.commission_per_share, self.exec.min_commission)

        # Create a minimal Fill object via OMS models
        from quant_strategies.execution.models import Fill
        fill = Fill.create(order=order, quantity=order.quantity, price=execution_price, fees=commission)
        return [fill]

    # Placeholders for OMS interface
    def get_quote(self, ticker: str): return None
    def cancel_order(self, order_id: str) -> bool: return False
    def get_order_status(self, order_id: str): return None


class EquityOMSBacktester:
    """
    Bar-by-bar equity backtester that replays OHLCV, generates strategy signals,
    routes through the OMS, and simulates fills at next open.
    """
    def __init__(self,
                 initial_cash: float = 100_000.0,
                 exec_config: Optional[ExecConfig] = None,
                 ledger_path: str = "data/equity_oms_backtest.db"):
        self.initial_cash = initial_cash
        self.exec_config = exec_config or ExecConfig()
        self.ledger_path = ledger_path

    def _fetch_hist(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Handle new yfinance format with multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns - use the first level (Price names)
            df.columns = df.columns.get_level_values(0)
        
        # Normalize columns to lower-case
        df = df.rename(columns=str.lower)
        # Ensure required columns: open, high, low, close, volume
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return pd.DataFrame()
        df = df.dropna(subset=["open", "high", "low", "close"])  # allow volume NaN
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def _build_market_data(self, bar_next_open: float) -> Dict:
        # Provide fields expected by OMS Broker (we override with next_open)
        return {
            "last": bar_next_open,
            "bid": bar_next_open * 0.999,
            "ask": bar_next_open * 1.001,
            "close": bar_next_open,
            "volume": 0,
            "timestamp": datetime.now(),
            "next_open": bar_next_open,
        }

    def run(self,
            tickers: List[str],
            start: str,
            end: str,
            strategy: Optional[BollingerRSIStrategy] = None,
            lookback_bars: int = 100,
            interval: str = "1d",
            save_equity_csv: Optional[str] = None,
            save_trades_csv: Optional[str] = None,
            save_equity_png: Optional[str] = None) -> Dict:

        # Initialize OMS components
        portfolio = Portfolio(initial_cash=self.initial_cash)
        ledger = Ledger(db_path=self.ledger_path)
        broker = BacktestBrokerAgent(self.exec_config)
        sizer = PositionSizer()
        oms = OrderManagementSystem(portfolio, ledger, broker, sizer)

        # Strategy
        strategy = strategy or BollingerRSIStrategy()

        # Fetch data per ticker
        hist_map: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = self._fetch_hist(t, start, end, interval=interval)
            if df.empty:
                continue
            hist_map[t] = df

        # Align all tickers on common index
        all_dates = sorted(set().union(*[df.index for df in hist_map.values()])) if hist_map else []
        if not all_dates:
            return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "ledger_path": self.ledger_path}

        # Equity curve tracking
        equity_records = []

        # Loop over time
        for i, dt in enumerate(all_dates):
            # For each ticker, generate signal using history up to current dt (need at least lookback_bars)
            for t, df in hist_map.items():
                # Need next bar open to execute (skip last available bar)
                if dt not in df.index:
                    continue
                row_idx = df.index.get_loc(dt)
                if row_idx >= len(df.index) - 1:
                    # no next bar to execute on
                    continue

                # Build rolling window
                start_idx = max(0, row_idx - lookback_bars + 1)
                window = df.iloc[start_idx:row_idx + 1].copy()
                window = window.rename_axis("date").reset_index()
                # Strategy expects columns: open, high, low, close, volume
                if not strategy.validate_data(window):
                    continue

                # Indicators and signals
                with_ind = strategy.calculate_indicators(window)
                with_sig = strategy.generate_signals(with_ind)
                latest_signal = with_sig.iloc[-1]["signal"]

                # Map to strategy enum
                if hasattr(latest_signal, "value"):
                    s_val = latest_signal.value
                else:
                    s_val = str(latest_signal)

                if s_val not in ("BUY", "SELL"):
                    continue

                # Emit OMS signal
                from quant_strategies.strategies.base_strategy import SignalType as StrategySignalType
                st_signal_type = StrategySignalType.BUY if s_val == "BUY" else StrategySignalType.SELL
                strength = float(abs(with_sig.iloc[-1].get("signal_strength", 0.75)) or 0.75)
                oms_signal = strategy.emit_signal(t, st_signal_type, strength=strength)

                # Next bar open defines execution price
                next_open = float(df["open"].iloc[row_idx + 1])
                mkt = self._build_market_data(next_open)
                oms.process_signal(oms_signal, mkt)

            # After processing all tickers at dt, record equity
            # Build current price snapshot for held positions (use current close)
            current_prices: Dict[str, float] = {}
            for t, df in hist_map.items():
                if dt in df.index:
                    current_prices[t] = float(df.loc[dt, "close"])
            total_val = portfolio.get_total_value(current_prices)
            unreal, real, total = portfolio.calculate_pnl(current_prices)

            equity_records.append({
                "timestamp": dt,
                "cash": portfolio.cash,
                "unrealized_pnl": unreal,
                "realized_pnl": real,
                "total_pnl": total,
                "total_value": total_val,
            })

        equity_curve = pd.DataFrame(equity_records).set_index("timestamp")

        # Optional saves
        if save_equity_csv:
            os.makedirs(os.path.dirname(save_equity_csv), exist_ok=True)
            equity_curve.to_csv(save_equity_csv)

        # Extract simple trades view from ledger
        import sqlite3
        conn = sqlite3.connect(self.ledger_path)
        trades_df = pd.read_sql_query(
            """
            SELECT o.order_id, o.signal_id, o.ticker, o.side, o.quantity, o.status, o.updated_at
            FROM orders o
            WHERE o.status='FILLED'
            """,
            conn,
        )
        conn.close()

        if save_trades_csv:
            os.makedirs(os.path.dirname(save_trades_csv), exist_ok=True)
            trades_df.to_csv(save_trades_csv, index=False)

        # Optional plot
        if save_equity_png:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                equity_curve["total_value"].plot(ax=ax)
                ax.set_title("Equity Curve (Total Portfolio Value)")
                ax.set_ylabel("Value ($)")
                ax.grid(True, alpha=0.3)
                os.makedirs(os.path.dirname(save_equity_png), exist_ok=True)
                plt.tight_layout()
                plt.savefig(save_equity_png)
                plt.close(fig)
            except Exception:
                pass

        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "ledger_path": self.ledger_path,
        }
