"""
Alpha Vantage FTSE 100 Data Provider

This module provides FTSE 100 data using the Alpha Vantage API,
with the same interface as the Intrinio provider for seamless integration.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
from dataclasses import dataclass
import logging
import requests
import time

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class IndexSnapshot:
    """Container for index snapshot data."""
    timestamp: pd.Timestamp
    index_px: float
    volume: Optional[int] = None


@dataclass
class OptionSnapshot:
    """Container for option snapshot data."""
    timestamp: pd.Timestamp
    cp: str  # call/put
    strike: float
    expiry: pd.Timestamp
    bid: float
    ask: float
    mid: float
    oi: int
    volume: Optional[int] = None
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class AlphaVantageFTSEProvider:
    """
    Alpha Vantage API client for FTSE 100 data.
    
    Provides methods to fetch index data with the same interface
    as the Intrinio provider for seamless integration.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage FTSE provider.
        
        Args:
            api_key: Alpha Vantage API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        
        # London timezone
        self.london_tz = pytz.timezone('Europe/London')
        
        # FTSE 100 symbol for Alpha Vantage
        self.ftse_symbol = "FTSE"  # Alpha Vantage uses "FTSE" for FTSE 100
        
        # Rate limiting (5 calls per minute for free tier)
        self.last_call_time = 0
        self.min_call_interval = 12  # 12 seconds between calls (5 per minute)
        
    def _rate_limit(self):
        """Apply rate limiting for Alpha Vantage API."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_request(self, params: Dict) -> Dict:
        """Make a rate-limited request to Alpha Vantage API."""
        self._rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API note: {data['Note']}")
            
            if 'Information' in data:
                logger.info(f"Alpha Vantage API info: {data['Information']}")
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to make Alpha Vantage request: {e}")
            raise
    
    def get_index_snapshot(self, 
                          date: str, 
                          near_time: str = "15:45", 
                          window: int = 20) -> IndexSnapshot:
        """
        Get FTSE 100 index snapshot near a specific time.
        
        Note: Alpha Vantage provides daily data, so we'll return
        the closing price for the specified date.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format (ignored for daily data)
            window: Window in minutes around target time (ignored for daily data)
            
        Returns:
            IndexSnapshot with timestamp and index price
        """
        logger.info(f"Fetching FTSE 100 snapshot for {date} from Alpha Vantage")
        
        try:
            # Parse date
            target_date = pd.to_datetime(date).date()
            
            # Get daily data for the month containing the target date
            start_date = target_date - timedelta(days=15)
            end_date = target_date + timedelta(days=15)
            
            # Fetch daily data
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': self.ftse_symbol,
                'outputsize': 'full'
            }
            
            data = self._make_request(params)
            
            if 'Time Series (Daily)' not in data:
                raise ValueError(f"No daily data found for {self.ftse_symbol}")
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'date': pd.to_datetime(date_str).date(),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('date')
            
            # Find the closest date to target
            if target_date in df['date'].values:
                closest_data = df[df['date'] == target_date].iloc[0]
            else:
                # Find the closest available date
                df['date_diff'] = abs((df['date'] - target_date).dt.days)
                closest_data = df.loc[df['date_diff'].idxmin()]
                logger.warning(f"Using data from {closest_data['date']} (closest to {target_date})")
            
            # Create snapshot
            # Use London timezone for the timestamp
            timestamp = pd.Timestamp.combine(target_date, datetime.strptime(near_time, "%H:%M").time())
            timestamp = self.london_tz.localize(timestamp)
            
            snapshot = IndexSnapshot(
                timestamp=timestamp,
                index_px=closest_data['close'],
                volume=closest_data['volume']
            )
            
            logger.info(f"Retrieved FTSE 100 data: {snapshot.index_px:.2f} at {snapshot.timestamp}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to fetch FTSE 100 data: {e}")
            raise
    
    def get_option_chain_snapshot(self, 
                                 date: str,
                                 near_time: str = "15:45",
                                 window: int = 20,
                                 maturity_bounds: Tuple[int, int] = (7, 60),
                                 spread_limit: float = 0.5,
                                 min_oi: int = 100) -> pd.DataFrame:
        """
        Get FTSE 100 options chain snapshot.
        
        Note: Alpha Vantage doesn't provide options data for FTSE 100.
        This method returns an empty DataFrame with a warning.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format
            window: Window in minutes around target time
            maturity_bounds: (min_days, max_days) for option maturity
            spread_limit: Maximum relative spread
            min_oi: Minimum open interest
            
        Returns:
            Empty DataFrame with warning message
        """
        logger.warning("Alpha Vantage doesn't provide FTSE 100 options data")
        logger.warning("Consider using Intrinio or another provider for options data")
        
        # Return empty DataFrame with proper structure
        columns = [
            'timestamp', 'cp', 'strike', 'expiry', 'bid', 'ask', 'mid', 'oi',
            'volume', 'iv', 'delta', 'gamma', 'theta', 'vega'
        ]
        
        return pd.DataFrame(columns=columns)
    
    def get_historical_data(self, 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """
        Get historical FTSE 100 data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching historical FTSE 100 data from {start_date} to {end_date}")
        
        try:
            # Fetch daily data
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': self.ftse_symbol,
                'outputsize': 'full'
            }
            
            data = self._make_request(params)
            
            if 'Time Series (Daily)' not in data:
                raise ValueError(f"No daily data found for {self.ftse_symbol}")
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('date')
            df.set_index('date', inplace=True)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            if df.empty:
                raise ValueError(f"No data found for {start_date} to {end_date}")
            
            # Convert to London timezone
            df.index = df.index.tz_localize('UTC').tz_convert(self.london_tz)
            
            logger.info(f"Retrieved {len(df)} days of historical data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    def get_intraday_data(self, 
                         symbol: str = None,
                         interval: str = "1min",
                         outputsize: str = "compact") -> pd.DataFrame:
        """
        Get intraday FTSE 100 data.
        
        Args:
            symbol: Symbol to fetch (defaults to FTSE)
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: compact (last 100) or full (all data)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        if symbol is None:
            symbol = self.ftse_symbol
        
        logger.info(f"Fetching intraday {symbol} data with {interval} interval")
        
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize
            }
            
            data = self._make_request(params)
            
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                raise ValueError(f"No intraday data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp_str, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            # Convert to London timezone
            df.index = df.index.tz_localize('UTC').tz_convert(self.london_tz)
            
            logger.info(f"Retrieved {len(df)} intraday records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch intraday data: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to fetch recent data
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': self.ftse_symbol,
                'outputsize': 'compact'
            }
            
            data = self._make_request(params)
            
            if 'Time Series (Daily)' in data:
                logger.info("Alpha Vantage API connection successful")
                return True
            else:
                logger.error("No data received from Alpha Vantage")
                return False
                
        except Exception as e:
            logger.error(f"Alpha Vantage API connection failed: {e}")
            return False


# Convenience functions for backward compatibility
def get_index_snapshot(date: str, 
                      near_time: str = "15:45", 
                      window: int = 20,
                      api_key: Optional[str] = None) -> IndexSnapshot:
    """
    Convenience function to get FTSE 100 index snapshot from Alpha Vantage.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        api_key: Alpha Vantage API key (optional if set in environment)
        
    Returns:
        IndexSnapshot with timestamp and index price
    """
    provider = AlphaVantageFTSEProvider(api_key)
    return provider.get_index_snapshot(date, near_time, window)


def get_historical_data(start_date: str, end_date: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to get historical FTSE 100 data from Alpha Vantage.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Alpha Vantage API key (optional if set in environment)
        
    Returns:
        DataFrame with OHLCV data
    """
    provider = AlphaVantageFTSEProvider(api_key)
    return provider.get_historical_data(start_date, end_date)
