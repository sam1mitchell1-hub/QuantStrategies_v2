import os
import pandas as pd
from datetime import datetime
from quant_strategies.data import YFinanceClient


def fake_download(ticker, start=None, end=None):
    # Minimal frame like yfinance output
    dates = pd.date_range(start=start, end=end, freq='D')
    if len(dates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({
        'Open': [100.0] * len(dates),
        'High': [101.0] * len(dates),
        'Low': [99.0] * len(dates),
        'Close': [100.5] * len(dates),
        'Adj Close': [100.5] * len(dates),
        'Volume': [1000] * len(dates),
    }, index=dates)
    return df


def test_get_daily_prices_monkeypatch(monkeypatch, tmp_path):
    monkeypatch.setattr("yfinance.download", fake_download)
    cli = YFinanceClient()
    df = cli.get_daily_prices("AAPL", "2020-01-01", "2020-01-05")
    assert not df.empty
    assert set(['date','open','high','low','close','adj_close','volume']).issubset(df.columns)


def test_save_daily_prices_from_start(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.download", fake_download)
    cli = YFinanceClient()
    out_dir = tmp_path / "out"
    path = cli.save_daily_prices_from_start("TSLA", "2020-01-01", output_dir=str(out_dir), filename="tsla.csv")
    assert os.path.exists(path)
    saved = pd.read_csv(path)
    assert 'date' in saved.columns
    assert len(saved) > 0
