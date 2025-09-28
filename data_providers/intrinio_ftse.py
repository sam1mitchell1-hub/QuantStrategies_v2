"""
Intrinio FTSE 100 Options Data Provider

This module provides functions to fetch FTSE 100 index data and options chain data
from the Intrinio API, with proper timezone handling and data filtering.
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
from dataclasses import dataclass
import logging

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


class IntrinioFTSEProvider:
    """
    Intrinio API client for FTSE 100 data.
    
    Provides methods to fetch index snapshots and options chain data
    with proper timezone handling and data filtering.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Intrinio FTSE provider.
        
        Args:
            api_key: Intrinio API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('INTRINIO_API_KEY')
        if not self.api_key:
            raise ValueError("Intrinio API key not provided. Set INTRINIO_API_KEY environment variable.")
        
        self.base_url = "https://api-v2.intrinio.com"
        self.session = requests.Session()
        self.session.auth = (self.api_key, '')
        
        # London timezone
        self.london_tz = pytz.timezone('Europe/London')
        
        # FTSE 100 identifier
        self.ftse_symbol = "UKX"
        
    def get_index_snapshot(self, 
                          date: str, 
                          near_time: str = "15:45", 
                          window: int = 20) -> IndexSnapshot:
        """
        Get FTSE 100 index snapshot near a specific time.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format (default: "15:45")
            window: Window in minutes around target time (default: 20)
            
        Returns:
            IndexSnapshot with timestamp and index price
            
        Raises:
            ValueError: If no data found in time window
            requests.RequestException: If API request fails
        """
        logger.info(f"Fetching FTSE 100 snapshot for {date} near {near_time}")
        
        # Parse date and time
        target_date = pd.to_datetime(date).date()
        target_time = datetime.strptime(near_time, "%H:%M").time()
        target_datetime = datetime.combine(target_date, target_time)
        target_datetime_london = self.london_tz.localize(target_datetime)
        
        # Calculate time window
        window_start = target_datetime_london - timedelta(minutes=window)
        window_end = target_datetime_london + timedelta(minutes=window)
        
        # Convert to UTC for API request
        window_start_utc = window_start.astimezone(pytz.UTC)
        window_end_utc = window_end.astimezone(pytz.UTC)
        
        # Fetch intraday data
        intraday_data = self._fetch_intraday_data(
            symbol=self.ftse_symbol,
            start_time=window_start_utc,
            end_time=window_end_utc,
            frequency="1min"
        )
        
        if intraday_data.empty:
            raise ValueError(f"No intraday data found for {date} in time window {near_time}±{window}m")
        
        # Convert timestamps to London timezone
        intraday_data['timestamp_london'] = intraday_data['timestamp'].dt.tz_convert(self.london_tz)
        
        # Find closest record to target time
        time_diffs = abs(intraday_data['timestamp_london'] - target_datetime_london)
        closest_idx = time_diffs.idxmin()
        closest_record = intraday_data.loc[closest_idx]
        
        logger.info(f"Found closest record at {closest_record['timestamp_london']} "
                   f"(diff: {time_diffs[closest_idx].total_seconds()/60:.1f} minutes)")
        
        return IndexSnapshot(
            timestamp=closest_record['timestamp_london'],
            index_px=closest_record['close'],
            volume=closest_record.get('volume')
        )
    
    def get_option_chain_snapshot(self, 
                                 date: str,
                                 near_time: str = "15:45",
                                 window: int = 20,
                                 maturity_bounds: Tuple[int, int] = (7, 60),
                                 spread_limit: float = 0.5,
                                 min_oi: int = 100) -> pd.DataFrame:
        """
        Get FTSE 100 options chain snapshot near a specific time.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format (default: "15:45")
            window: Window in minutes around target time (default: 20)
            maturity_bounds: (min_days, max_days) for option maturity
            spread_limit: Maximum relative spread (ask-bid)/mid
            min_oi: Minimum open interest
            
        Returns:
            DataFrame with option data columns:
            [timestamp, cp, strike, expiry, bid, ask, mid, oi, volume, iv, delta, gamma, theta, vega]
            
        Raises:
            ValueError: If no data found in time window
            requests.RequestException: If API request fails
        """
        logger.info(f"Fetching FTSE 100 options chain for {date} near {near_time}")
        
        # Parse date and time
        target_date = pd.to_datetime(date).date()
        target_time = datetime.strptime(near_time, "%H:%M").time()
        target_datetime = datetime.combine(target_date, target_time)
        target_datetime_london = self.london_tz.localize(target_datetime)
        
        # Calculate time window
        window_start = target_datetime_london - timedelta(minutes=window)
        window_end = target_datetime_london + timedelta(minutes=window)
        
        # Convert to UTC for API request
        window_start_utc = window_start.astimezone(pytz.UTC)
        window_end_utc = window_end.astimezone(pytz.UTC)
        
        # Fetch options data
        options_data = self._fetch_options_data(
            underlying=self.ftse_symbol,
            start_time=window_start_utc,
            end_time=window_end_utc
        )
        
        if options_data.empty:
            raise ValueError(f"No options data found for {date} in time window {near_time}±{window}m")
        
        # Convert timestamps to London timezone
        options_data['timestamp_london'] = options_data['timestamp'].dt.tz_convert(self.london_tz)
        
        # Filter to near-EOD window
        mask = (options_data['timestamp_london'] >= window_start) & (options_data['timestamp_london'] <= window_end)
        window_data = options_data[mask].copy()
        
        if window_data.empty:
            raise ValueError(f"No options data found in time window {near_time}±{window}m")
        
        # Find closest timestamp to target time
        time_diffs = abs(window_data['timestamp_london'] - target_datetime_london)
        closest_timestamp = time_diffs.idxmin()
        snapshot_data = window_data.loc[closest_timestamp]
        
        logger.info(f"Found options snapshot at {snapshot_data['timestamp_london']} "
                   f"(diff: {time_diffs[closest_timestamp].total_seconds()/60:.1f} minutes)")
        
        # Process and filter options data
        processed_data = self._process_options_data(
            snapshot_data,
            maturity_bounds,
            spread_limit,
            min_oi
        )
        
        return processed_data
    
    def _fetch_intraday_data(self, 
                           symbol: str,
                           start_time: datetime,
                           end_time: datetime,
                           frequency: str = "1min") -> pd.DataFrame:
        """Fetch intraday data from Intrinio API."""
        url = f"{self.base_url}/securities/{symbol}/prices/intraday"
        
        params = {
            'start_date': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'end_date': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'frequency': frequency,
            'page_size': 1000
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('intraday_prices'):
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['intraday_prices'])
            df['timestamp'] = pd.to_datetime(df['time'])
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(int)
            
            return df[['timestamp', 'close', 'volume']]
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch intraday data: {e}")
            raise
    
    def _fetch_options_data(self, 
                          underlying: str,
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Fetch options data from Intrinio API with pagination."""
        url = f"{self.base_url}/options/prices"
        
        all_data = []
        page = 1
        page_size = 1000
        
        while True:
            params = {
                'underlying_symbol': underlying,
                'start_date': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'end_date': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'page_size': page_size,
                'page': page
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('options'):
                    break
                
                all_data.extend(data['options'])
                
                # Check if we have more pages
                if len(data['options']) < page_size:
                    break
                    
                page += 1
                
            except requests.RequestException as e:
                logger.error(f"Failed to fetch options data (page {page}): {e}")
                raise
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['date'])
        
        return df
    
    def _process_options_data(self, 
                            snapshot_data: pd.Series,
                            maturity_bounds: Tuple[int, int],
                            spread_limit: float,
                            min_oi: int) -> pd.DataFrame:
        """Process and filter options data."""
        # Convert to DataFrame if it's a Series
        if isinstance(snapshot_data, pd.Series):
            df = pd.DataFrame([snapshot_data])
        else:
            df = snapshot_data.copy()
        
        # Ensure we have the required columns
        required_cols = ['cp', 'strike', 'expiry', 'bid', 'ask', 'oi']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert data types
        df['strike'] = df['strike'].astype(float)
        df['bid'] = df['bid'].astype(float)
        df['ask'] = df['ask'].astype(float)
        df['oi'] = df['oi'].astype(int)
        df['expiry'] = pd.to_datetime(df['expiry'])
        
        # Calculate mid price and relative spread
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['rel_spread'] = (df['ask'] - df['bid']) / df['mid']
        
        # Filter data
        min_days, max_days = maturity_bounds
        current_date = df['timestamp_london'].iloc[0].date()
        
        # Maturity filter
        df['days_to_expiry'] = (df['expiry'].dt.date - current_date).dt.days
        maturity_mask = (df['days_to_expiry'] >= min_days) & (df['days_to_expiry'] <= max_days)
        
        # Quality filters
        quality_mask = (
            (df['bid'] > 0) &
            (df['ask'] > df['bid']) &
            (df['rel_spread'] <= spread_limit) &
            (df['oi'] >= min_oi)
        )
        
        # Apply filters
        filtered_df = df[maturity_mask & quality_mask].copy()
        
        # Add optional columns if available
        optional_cols = ['volume', 'iv', 'delta', 'gamma', 'theta', 'vega']
        for col in optional_cols:
            if col in df.columns:
                filtered_df[col] = df[col]
            else:
                filtered_df[col] = None
        
        # Select and order columns
        output_cols = [
            'timestamp_london', 'cp', 'strike', 'expiry', 'bid', 'ask', 'mid', 'oi',
            'volume', 'iv', 'delta', 'gamma', 'theta', 'vega'
        ]
        
        result_df = filtered_df[output_cols].copy()
        result_df = result_df.rename(columns={'timestamp_london': 'timestamp'})
        
        logger.info(f"Processed {len(filtered_df)} options from {len(df)} total records")
        
        return result_df
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = f"{self.base_url}/securities/{self.ftse_symbol}"
            response = self.session.get(url)
            response.raise_for_status()
            logger.info("Intrinio API connection successful")
            return True
        except Exception as e:
            logger.error(f"Intrinio API connection failed: {e}")
            return False


# Convenience functions for backward compatibility
def get_index_snapshot(date: str, 
                      near_time: str = "15:45", 
                      window: int = 20,
                      api_key: Optional[str] = None) -> IndexSnapshot:
    """
    Convenience function to get FTSE 100 index snapshot.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        api_key: Intrinio API key (optional if set in environment)
        
    Returns:
        IndexSnapshot with timestamp and index price
    """
    provider = IntrinioFTSEProvider(api_key)
    return provider.get_index_snapshot(date, near_time, window)


def get_option_chain_snapshot(date: str,
                             near_time: str = "15:45",
                             window: int = 20,
                             maturity_bounds: Tuple[int, int] = (7, 60),
                             spread_limit: float = 0.5,
                             min_oi: int = 100,
                             api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to get FTSE 100 options chain snapshot.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        maturity_bounds: (min_days, max_days) for option maturity
        spread_limit: Maximum relative spread
        min_oi: Minimum open interest
        api_key: Intrinio API key (optional if set in environment)
        
    Returns:
        DataFrame with option data
    """
    provider = IntrinioFTSEProvider(api_key)
    return provider.get_option_chain_snapshot(
        date, near_time, window, maturity_bounds, spread_limit, min_oi
    )
