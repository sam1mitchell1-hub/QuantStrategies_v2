"""
Sample FTSE 100 Data Provider

This module provides realistic sample data for FTSE 100 index and options
when real data sources are not available. Useful for development and testing.
"""

import pandas as pd
import numpy as np
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


class SampleFTSEProvider:
    """
    Sample data provider for FTSE 100.
    
    Generates realistic sample data for development and testing
    when real data sources are not available.
    """
    
    def __init__(self, base_price: float = 7500.0, volatility: float = 0.20):
        """
        Initialize sample FTSE provider.
        
        Args:
            base_price: Base FTSE 100 price for generating data
            volatility: Volatility for generating realistic price movements
        """
        self.base_price = base_price
        self.volatility = volatility
        self.london_tz = pytz.timezone('Europe/London')
        
        # Set random seed for reproducible data
        np.random.seed(42)
        
        logger.info(f"Sample FTSE provider initialized with base price {base_price}")
    
    def get_index_snapshot(self, 
                          date: str, 
                          near_time: str = "15:45", 
                          window: int = 20) -> IndexSnapshot:
        """
        Get sample FTSE 100 index snapshot.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format
            window: Window in minutes around target time
            
        Returns:
            IndexSnapshot with sample data
        """
        logger.info(f"Generating sample FTSE 100 snapshot for {date}")
        
        # Parse date
        target_date = pd.to_datetime(date).date()
        
        # Generate realistic price based on date
        days_since_epoch = (target_date - datetime(2020, 1, 1).date()).days
        price_trend = self.base_price * (1 + 0.0001 * days_since_epoch)  # Slight upward trend
        
        # Add some random variation
        daily_return = np.random.normal(0, self.volatility / np.sqrt(252))
        price = price_trend * (1 + daily_return)
        
        # Generate volume
        volume = np.random.randint(1000000, 5000000)
        
        # Create timestamp
        target_time = datetime.strptime(near_time, "%H:%M").time()
        timestamp = pd.Timestamp.combine(target_date, target_time)
        timestamp = self.london_tz.localize(timestamp)
        
        snapshot = IndexSnapshot(
            timestamp=timestamp,
            index_px=round(price, 2),
            volume=volume
        )
        
        logger.info(f"Generated sample data: {snapshot.index_px:.2f} at {snapshot.timestamp}")
        return snapshot
    
    def get_option_chain_snapshot(self, 
                                 date: str,
                                 near_time: str = "15:45",
                                 window: int = 20,
                                 maturity_bounds: Tuple[int, int] = (7, 60),
                                 spread_limit: float = 0.5,
                                 min_oi: int = 100) -> pd.DataFrame:
        """
        Get sample FTSE 100 options chain snapshot.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format
            window: Window in minutes around target time
            maturity_bounds: (min_days, max_days) for option maturity
            spread_limit: Maximum relative spread
            min_oi: Minimum open interest
            
        Returns:
            DataFrame with sample options data
        """
        logger.info(f"Generating sample FTSE 100 options chain for {date}")
        
        # Get current index price
        index_snapshot = self.get_index_snapshot(date, near_time, window)
        current_price = index_snapshot.index_px
        
        # Generate strikes around current price
        min_days, max_days = maturity_bounds
        num_strikes = 20
        strike_range = 0.4  # 40% around current price
        
        strikes = np.linspace(
            current_price * (1 - strike_range/2),
            current_price * (1 + strike_range/2),
            num_strikes
        )
        
        # Generate expiries
        num_expiries = 4
        expiries = []
        for i in range(num_expiries):
            days_to_expiry = min_days + (max_days - min_days) * i / (num_expiries - 1)
            expiry_date = pd.to_datetime(date) + pd.Timedelta(days=days_to_expiry)
            expiries.append(expiry_date)
        
        # Generate options data
        options = []
        timestamp = index_snapshot.timestamp
        
        for strike in strikes:
            for expiry in expiries:
                days_to_expiry = (expiry.date() - pd.to_datetime(date).date()).days
                
                # Generate realistic option prices using Black-Scholes approximation
                call_price, put_price, iv = self._generate_option_prices(
                    current_price, strike, days_to_expiry, self.volatility
                )
                
                # Generate Greeks
                delta_call, gamma, theta_call, vega = self._calculate_greeks(
                    current_price, strike, days_to_expiry, self.volatility, 'call'
                )
                delta_put = delta_call - 1
                theta_put = theta_call + current_price * 0.01  # Rough approximation
                
                # Generate bid-ask spreads
                call_spread = call_price * 0.02  # 2% spread
                put_spread = put_price * 0.02
                
                # Generate open interest and volume
                oi = np.random.randint(min_oi, 1000)
                volume = np.random.randint(0, 100)
                
                # Call option
                options.append({
                    'timestamp': timestamp,
                    'cp': 'call',
                    'strike': round(strike, 2),
                    'expiry': expiry,
                    'bid': round(call_price - call_spread/2, 2),
                    'ask': round(call_price + call_spread/2, 2),
                    'mid': round(call_price, 2),
                    'oi': oi,
                    'volume': volume,
                    'iv': round(iv, 4),
                    'delta': round(delta_call, 4),
                    'gamma': round(gamma, 4),
                    'theta': round(theta_call, 4),
                    'vega': round(vega, 4)
                })
                
                # Put option
                options.append({
                    'timestamp': timestamp,
                    'cp': 'put',
                    'strike': round(strike, 2),
                    'expiry': expiry,
                    'bid': round(put_price - put_spread/2, 2),
                    'ask': round(put_price + put_spread/2, 2),
                    'mid': round(put_price, 2),
                    'oi': oi,
                    'volume': volume,
                    'iv': round(iv, 4),
                    'delta': round(delta_put, 4),
                    'gamma': round(gamma, 4),
                    'theta': round(theta_put, 4),
                    'vega': round(vega, 4)
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(options)
        
        # Apply filters
        df['rel_spread'] = (df['ask'] - df['bid']) / df['mid']
        df['days_to_expiry'] = (df['expiry'].dt.date - pd.to_datetime(date).date()).dt.days
        
        # Filter by maturity bounds
        maturity_mask = (df['days_to_expiry'] >= min_days) & (df['days_to_expiry'] <= max_days)
        
        # Filter by quality
        quality_mask = (
            (df['bid'] > 0) &
            (df['ask'] > df['bid']) &
            (df['rel_spread'] <= spread_limit) &
            (df['oi'] >= min_oi)
        )
        
        # Apply filters
        filtered_df = df[maturity_mask & quality_mask].copy()
        
        # Remove helper columns
        filtered_df = filtered_df.drop(['rel_spread', 'days_to_expiry'], axis=1)
        
        logger.info(f"Generated {len(filtered_df)} sample options from {len(df)} total")
        return filtered_df
    
    def get_historical_data(self, 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """
        Get sample historical FTSE 100 data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with sample OHLCV data
        """
        logger.info(f"Generating sample historical data from {start_date} to {end_date}")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price series with realistic movements
        returns = np.random.normal(0, self.volatility / np.sqrt(252), len(dates))
        prices = [self.base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            daily_vol = self.volatility / np.sqrt(252)
            high_factor = 1 + abs(np.random.normal(0, daily_vol * 0.5))
            low_factor = 1 - abs(np.random.normal(0, daily_vol * 0.5))
            
            open_price = price * (1 + np.random.normal(0, daily_vol * 0.3))
            high_price = max(open_price, price) * high_factor
            low_price = min(open_price, price) * low_factor
            close_price = price
            
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index = df.index.tz_localize('UTC').tz_convert(self.london_tz)
        
        logger.info(f"Generated {len(df)} days of sample historical data")
        return df
    
    def _generate_option_prices(self, S, K, T, sigma, r=0.05):
        """Generate realistic option prices using Black-Scholes approximation."""
        from scipy.stats import norm
        
        if T <= 0:
            # At expiry
            call_price = max(0, S - K)
            put_price = max(0, K - S)
            iv = 0.0
        else:
            # Black-Scholes approximation
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T/365) / (sigma*np.sqrt(T/365))
            d2 = d1 - sigma*np.sqrt(T/365)
            
            call_price = S*norm.cdf(d1) - K*np.exp(-r*T/365)*norm.cdf(d2)
            put_price = K*np.exp(-r*T/365)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
            # Add some randomness to IV
            iv = sigma * (1 + np.random.normal(0, 0.1))
        
        return max(0, call_price), max(0, put_price), iv
    
    def _calculate_greeks(self, S, K, T, sigma, option_type, r=0.05):
        """Calculate option Greeks."""
        from scipy.stats import norm
        
        if T <= 0:
            return 0, 0, 0, 0
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T/365) / (sigma*np.sqrt(T/365))
        d2 = d1 - sigma*np.sqrt(T/365)
        
        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T/365))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T/365)) - r * K * np.exp(-r*T/365) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T/365)
        
        return delta, gamma, theta, vega
    
    def test_connection(self) -> bool:
        """
        Test sample data generation.
        
        Returns:
            True if sample data can be generated
        """
        try:
            # Test with recent date
            test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            snapshot = self.get_index_snapshot(test_date)
            
            if snapshot.index_px > 0:
                logger.info("Sample data generation successful")
                return True
            else:
                logger.error("Sample data generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Sample data generation failed: {e}")
            return False


# Convenience functions
def get_index_snapshot(date: str, 
                      near_time: str = "15:45", 
                      window: int = 20) -> IndexSnapshot:
    """
    Convenience function to get sample FTSE 100 index snapshot.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        
    Returns:
        IndexSnapshot with sample data
    """
    provider = SampleFTSEProvider()
    return provider.get_index_snapshot(date, near_time, window)


def get_option_chain_snapshot(date: str,
                             near_time: str = "15:45",
                             window: int = 20,
                             maturity_bounds: Tuple[int, int] = (7, 60),
                             spread_limit: float = 0.5,
                             min_oi: int = 100) -> pd.DataFrame:
    """
    Convenience function to get sample FTSE 100 options chain snapshot.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        maturity_bounds: (min_days, max_days) for option maturity
        spread_limit: Maximum relative spread
        min_oi: Minimum open interest
        
    Returns:
        DataFrame with sample options data
    """
    provider = SampleFTSEProvider()
    return provider.get_option_chain_snapshot(
        date, near_time, window, maturity_bounds, spread_limit, min_oi
    )


def get_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convenience function to get sample historical FTSE 100 data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with sample OHLCV data
    """
    provider = SampleFTSEProvider()
    return provider.get_historical_data(start_date, end_date)
