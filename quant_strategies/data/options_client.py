import os
from typing import List, Dict, Optional
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

class OptionsClient:
    def __init__(self, data_dir: str = "data/options", max_retries: int = 3, retry_delay_sec: float = 0.75):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.metadata_dir = os.path.join(self.data_dir, "metadata")
        for d in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    def _ticker(self, ticker: str) -> yf.Ticker:
        # Use yfinance's default session (required for curl_cffi pipeline)
        return yf.Ticker(ticker)

    def list_expirations(self, ticker: str) -> List[str]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                tk = self._ticker(ticker)
                opts = tk.options or []
                if opts:
                    return opts
            except Exception as exc:
                last_exc = exc
            time.sleep(self.retry_delay_sec)
        if last_exc:
            print(f"Failed to list expirations for {ticker} after {self.max_retries} retries: {last_exc}")
        return []

    def fetch_chain(self, ticker: str, expiry: str) -> Dict[str, pd.DataFrame]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                tk = self._ticker(ticker)
                chain = tk.option_chain(expiry)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                # Add metadata columns
                for df in (calls, puts):
                    df["ticker"] = ticker
                    df["expiry"] = expiry
                    if "lastTradeDate" in df.columns:
                        df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"], errors="coerce")
                return {"calls": calls, "puts": puts}
            except Exception as exc:
                last_exc = exc
                time.sleep(self.retry_delay_sec)
        raise RuntimeError(f"Failed to fetch option chain for {ticker} {expiry} after {self.max_retries} retries: {last_exc}")

    def fetch_and_save(self, ticker: str, expiries: Optional[List[str]] = None) -> Dict[str, List[str]]:
        if expiries is None:
            expiries = self.list_expirations(ticker)
        if not expiries:
            raise ValueError(f"No expirations found for {ticker}")

        saved = {"calls": [], "puts": []}
        for expiry in expiries:
            try:
                chain = self.fetch_chain(ticker, expiry)
                # Save files
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = f"{ticker}_{expiry}_{ts}"
                calls_path = os.path.join(self.raw_dir, f"{base}_calls.csv")
                puts_path = os.path.join(self.raw_dir, f"{base}_puts.csv")
                chain["calls"].to_csv(calls_path, index=False)
                chain["puts"].to_csv(puts_path, index=False)
                saved["calls"].append(calls_path)
                saved["puts"].append(puts_path)
                print(f"Saved {ticker} {expiry}: calls->{calls_path}, puts->{puts_path}")
            except Exception as e:
                print(f"Error fetching {ticker} {expiry}: {e}")
            time.sleep(self.retry_delay_sec)
        return saved

    def fetch_near_expiries(self, ticker: str, num_expiries: int = 3) -> Dict[str, List[str]]:
        expiries = self.list_expirations(ticker)[:num_expiries]
        return self.fetch_and_save(ticker, expiries)
