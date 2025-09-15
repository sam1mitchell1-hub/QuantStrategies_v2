import os
import types
import pandas as pd
from datetime import datetime
from quant_strategies.data.options_client import OptionsClient

class FakeChain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame):
        self.calls = calls
        self.puts = puts

class FakeTicker:
    def __init__(self, options_list, chain_map):
        self._options = options_list
        self._chain_map = chain_map
    
    @property
    def options(self):
        return self._options
    
    def option_chain(self, expiry):
        calls, puts = self._chain_map[expiry]
        return FakeChain(calls, puts)


def test_options_client_fetch_and_save(monkeypatch, tmp_path):
    # Prepare fake data
    expiry = "2025-09-12"
    calls_df = pd.DataFrame({
        'contractSymbol': ['SPY250912C00600000'],
        'lastTradeDate': [pd.Timestamp('2025-09-10')],
        'strike': [600.0],
        'lastPrice': [10.0],
        'bid': [9.8],
        'ask': [10.2],
        'impliedVolatility': [0.25],
        'volume': [100],
        'openInterest': [200],
    })
    puts_df = pd.DataFrame({
        'contractSymbol': ['SPY250912P00600000'],
        'lastTradeDate': [pd.Timestamp('2025-09-10')],
        'strike': [600.0],
        'lastPrice': [9.5],
        'bid': [9.2],
        'ask': [9.8],
        'impliedVolatility': [0.26],
        'volume': [120],
        'openInterest': [180],
    })

    options_list = [expiry]
    chain_map = {expiry: (calls_df, puts_df)}

    # Monkeypatch yfinance.Ticker to return FakeTicker
    import yfinance as yf
    def fake_ticker(symbol):
        return FakeTicker(options_list, chain_map)
    monkeypatch.setattr(yf, "Ticker", fake_ticker)

    client = OptionsClient(data_dir=str(tmp_path))
    saved = client.fetch_near_expiries("SPY", num_expiries=1)

    assert len(saved['calls']) == 1
    assert len(saved['puts']) == 1
    assert os.path.exists(saved['calls'][0])
    assert os.path.exists(saved['puts'][0])

    # Validate saved content includes our metadata additions
    saved_calls = pd.read_csv(saved['calls'][0])
    assert 'ticker' in saved_calls.columns and 'expiry' in saved_calls.columns
    assert saved_calls.loc[0, 'ticker'] == 'SPY'
    assert saved_calls.loc[0, 'expiry'] == expiry
