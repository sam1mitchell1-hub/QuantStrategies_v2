import json
import os
from datetime import datetime
from quant_strategies.data.data_manager import DataManager

class DummyClient:
    def __init__(self, out_dir):
        self.out_dir = out_dir
    def save_daily_prices_from_start(self, ticker, start_date, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            f.write('date,open,high,low,close,volume\n')
            f.write('2020-01-01,1,1,1,1,100\n')
        return path
    def save_daily_prices(self, ticker, start_date, end_date, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            f.write('date,open,high,low,close,volume\n')
            f.write('2020-01-01,1,1,1,1,100\n')
        return path


def test_data_manager_multiple_tickers(tmp_path, monkeypatch):
    dm = DataManager(data_dir=str(tmp_path / 'data'))
    # replace real client with dummy
    dm.client = DummyClient(dm.raw_data_dir)

    tickers = ['AAPL','TSLA']
    res = dm.get_multiple_tickers(tickers, '2020-01-01')
    assert set(res.keys()) == set(tickers)
    for t in tickers:
        assert res[t] is not None
        assert os.path.exists(res[t])

    # metadata file exists
    meta_dir = os.path.join(str(tmp_path / 'data'), 'metadata')
    metas = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    assert metas, 'metadata json not created'
    # validate content keys
    with open(os.path.join(meta_dir, metas[0]), 'r') as f:
        meta = json.load(f)
    assert meta['tickers'] == tickers
    assert meta['start_date'] == '2020-01-01'


def test_data_manager_range(tmp_path, monkeypatch):
    dm = DataManager(data_dir=str(tmp_path / 'data'))
    dm.client = DummyClient(dm.raw_data_dir)
    res = dm.get_multiple_tickers_range(['MSFT'], '2020-01-01', '2020-12-31')
    assert 'MSFT' in res and res['MSFT'] is not None
    assert os.path.exists(res['MSFT'])
