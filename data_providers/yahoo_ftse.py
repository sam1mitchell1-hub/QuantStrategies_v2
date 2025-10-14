"""
Yahoo Finance FTSE 100 Data Provider

Fallback data provider using Yahoo Finance for FTSE 100 data
when Intrinio is not available or doesn't have the required access level.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
from dataclasses import dataclass
import logging
import yfinance as yf

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


class YahooFTSEProvider:
    """
    Yahoo Finance data provider for FTSE 100 data.
    
    This is a fallback provider when Intrinio is not available.
    Note: Yahoo Finance doesn't provide options data, so this is
    primarily for index data and basic market information.
    """
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        # Try different FTSE 100 symbols
        self.ftse_symbols = ["^FTSE", "FTSE.L", "UKX.L", "FTSE100.L"]
        self.ftse_symbol = None
        self.london_tz = pytz.timezone('Europe/London')
        
        # Find working symbol
        self._find_working_symbol()
    
    def _find_working_symbol(self):
        """Find a working FTSE 100 symbol."""
        for symbol in self.ftse_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Try to get recent data
                data = ticker.history(period="5d")
                if not data.empty:
                    self.ftse_symbol = symbol
                    logger.info(f"Using FTSE 100 symbol: {symbol}")
                    return
            except Exception as e:
                logger.debug(f"Symbol {symbol} not working: {e}")
                continue
        
        # If no symbol works, use the first one as fallback
        self.ftse_symbol = self.ftse_symbols[0]
        logger.warning(f"No working FTSE symbol found, using fallback: {self.ftse_symbol}")
        
    def get_index_snapshot(self, 
                          date: str, 
                          near_time: str = "15:45", 
                          window: int = 20) -> IndexSnapshot:
        """
        Get FTSE 100 index snapshot near a specific time.
        
        Note: Yahoo Finance provides daily data, so we'll return
        the closing price for the specified date.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format (ignored for daily data)
            window: Window in minutes around target time (ignored for daily data)
            
        Returns:
            IndexSnapshot with timestamp and index price
        """
        logger.info(f"Fetching FTSE 100 snapshot for {date} from Yahoo Finance")
        
        try:
            # Parse date
            target_date = pd.to_datetime(date).date()
            
            # Get data for the date and a few days around it
            start_date = target_date - timedelta(days=5)
            end_date = target_date + timedelta(days=5)
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(self.ftse_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {date}")
            
            # Find the closest date to target
            data.index = data.index.date
            if target_date in data.index:
                closest_data = data.loc[target_date]
            else:
                # Find the closest available date
                date_diffs = abs((data.index - target_date).days)
                closest_date = data.index[date_diffs.argmin()]
                closest_data = data.loc[closest_date]
                logger.warning(f"Using data from {closest_date} (closest to {target_date})")
            
            # Create snapshot
            # Use London timezone for the timestamp
            timestamp = pd.Timestamp.combine(target_date, datetime.strptime(near_time, "%H:%M").time())
            timestamp = self.london_tz.localize(timestamp)
            
            snapshot = IndexSnapshot(
                timestamp=timestamp,
                index_px=float(closest_data['Close']),
                volume=int(closest_data['Volume']) if 'Volume' in closest_data else None
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
        
        Note: Yahoo Finance doesn't provide options data for FTSE 100.
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
        logger.warning("Yahoo Finance doesn't provide FTSE 100 options data")
        logger.warning("Consider using Intrinio or another data provider for options data")
        
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
            ticker = yf.Ticker(self.ftse_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No historical data found for {start_date} to {end_date}")
            
            # Convert to London timezone
            data.index = data.index.tz_localize('UTC').tz_convert(self.london_tz)
            
            logger.info(f"Retrieved {len(data)} days of historical data")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test data connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to fetch recent data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            
            data = self.get_historical_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if not data.empty:
                logger.info("Yahoo Finance connection successful")
                return True
            else:
                logger.error("No data received from Yahoo Finance")
                return False
                
        except Exception as e:
            logger.error(f"Yahoo Finance connection failed: {e}")
            return False


# Convenience functions
def get_index_snapshot(date: str, 
                      near_time: str = "15:45", 
                      window: int = 20) -> IndexSnapshot:
    """
    Convenience function to get FTSE 100 index snapshot from Yahoo Finance.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        
    Returns:
        IndexSnapshot with timestamp and index price
    """
    provider = YahooFTSEProvider()
    return provider.get_index_snapshot(date, near_time, window)


def get_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convenience function to get historical FTSE 100 data from Yahoo Finance.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with OHLCV data
    """
    provider = YahooFTSEProvider()
    return provider.get_historical_data(start_date, end_date)
