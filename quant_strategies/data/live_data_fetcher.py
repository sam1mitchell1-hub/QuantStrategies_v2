"""
Live Data Fetcher

Fetches real-time price data for liquid stocks using yfinance.
Includes caching to avoid rate limits and provides OMS-compatible market data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


# List of highly liquid stocks
LIQUID_STOCKS = {
    # Large Cap Tech
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    
    # Finance
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corp.',
    'GS': 'Goldman Sachs Group Inc.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    
    # Consumer
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble Co.',
    'KO': 'Coca-Cola Co.',
    'PEP': 'PepsiCo Inc.',
    
    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth Group Inc.',
    'PFE': 'Pfizer Inc.',
}


class LiveDataFetcher:
    """
    Fetches real-time stock data using yfinance with caching.
    """
    
    def __init__(self, cache_duration_seconds: int = 60):
        """
        Initialize live data fetcher.
        
        Args:
            cache_duration_seconds: How long to cache price data (default 60s)
        """
        self.cache_duration = cache_duration_seconds
        self._price_cache: Dict[str, tuple] = {}  # {ticker: (data, timestamp)}
        self._hist_cache: Dict[str, tuple] = {}   # {ticker: (dataframe, timestamp)}
        
        logger.info(f"LiveDataFetcher initialized with {cache_duration_seconds}s cache")
    
    def get_liquid_stocks(self) -> List[str]:
        """
        Get list of liquid stock tickers.
        
        Returns:
            List of ticker symbols
        """
        return list(LIQUID_STOCKS.keys())
    
    def get_stock_name(self, ticker: str) -> str:
        """
        Get full company name for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company name or ticker if not found
        """
        return LIQUID_STOCKS.get(ticker, ticker)
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of {ticker: current_price}
        """
        prices = {}
        
        for ticker in tickers:
            try:
                market_data = self.get_market_data(ticker)
                if market_data and 'last' in market_data:
                    prices[ticker] = market_data['last']
                else:
                    logger.warning(f"No price data for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching price for {ticker}: {e}")
        
        return prices
    
    def get_market_data(self, ticker: str) -> Optional[Dict]:
        """
        Get current market data for a single ticker in OMS format.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with bid, ask, last, close, volume, timestamp
        """
        # Check cache
        if ticker in self._price_cache:
            cached_data, cached_time = self._price_cache[ticker]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                logger.debug(f"Using cached data for {ticker}")
                return cached_data
        
        try:
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            
            # Get current quote info
            info = stock.info
            
            # Get recent history for close price
            hist = stock.history(period="1d", interval="1m")
            
            if hist.empty:
                logger.warning(f"No historical data available for {ticker}")
                return None
            
            # Extract data
            last_price = float(hist['Close'].iloc[-1]) if not hist.empty else None
            volume = int(hist['Volume'].iloc[-1]) if not hist.empty else 0
            
            # Try to get bid/ask from info, fallback to estimates
            bid = info.get('bid', last_price * 0.999 if last_price else None)
            ask = info.get('ask', last_price * 1.001 if last_price else None)
            
            if last_price is None:
                logger.error(f"Could not determine price for {ticker}")
                return None
            
            market_data = {
                'ticker': ticker,
                'last': float(last_price),
                'bid': float(bid) if bid else float(last_price * 0.999),
                'ask': float(ask) if ask else float(last_price * 1.001),
                'close': float(last_price),
                'volume': volume,
                'timestamp': datetime.now()
            }
            
            # Cache the data
            self._price_cache[ticker] = (market_data, datetime.now())
            
            logger.debug(f"Fetched market data for {ticker}: ${last_price:.2f}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None
    
    def get_historical_data(self, ticker: str, period: str = "1y", 
                           interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data, or None if error
        """
        cache_key = f"{ticker}_{period}_{interval}"
        
        # Check cache (longer cache for historical data)
        if cache_key in self._hist_cache:
            cached_df, cached_time = self._hist_cache[cache_key]
            # Cache historical data for 5 minutes
            if (datetime.now() - cached_time).total_seconds() < 300:
                logger.debug(f"Using cached historical data for {ticker}")
                return cached_df.copy()
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No historical data for {ticker} (period={period})")
                return None
            
            # Rename columns to lowercase for consistency
            hist.columns = [col.lower() for col in hist.columns]
            
            # Cache the data
            self._hist_cache[cache_key] = (hist, datetime.now())
            
            logger.info(f"Fetched {len(hist)} bars of historical data for {ticker}")
            return hist.copy()
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None
    
    def get_latest_bars(self, ticker: str, n: int = 100, 
                       interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get the last N bars of historical data for a ticker.
        
        Useful for strategy indicators that need recent history.
        
        Args:
            ticker: Stock ticker symbol
            n: Number of bars to retrieve
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with last N bars of OHLCV data
        """
        # Determine period based on interval and n
        if interval in ['1d', '1wk']:
            # For daily data, get more than needed
            period = "1y" if n <= 252 else "2y"
        elif interval in ['1h', '60m']:
            period = "1mo"
        else:
            period = "5d"
        
        hist = self.get_historical_data(ticker, period=period, interval=interval)
        
        if hist is None or hist.empty:
            return None
        
        # Return last n bars
        return hist.tail(n).copy()
    
    def clear_cache(self):
        """Clear all cached data."""
        self._price_cache.clear()
        self._hist_cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        return {
            'price_cache_size': len(self._price_cache),
            'hist_cache_size': len(self._hist_cache),
            'cache_duration_seconds': self.cache_duration
        }

