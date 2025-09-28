"""
FTSE 100 Data Manager

This module provides a unified interface for FTSE 100 data from multiple sources,
with automatic fallback when primary sources are unavailable.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

# Import data providers
try:
    from .intrinio_ftse import IntrinioFTSEProvider, IndexSnapshot as IntrinioIndexSnapshot
    INTRINIO_AVAILABLE = True
except ImportError:
    INTRINIO_AVAILABLE = False

try:
    from .yahoo_ftse import YahooFTSEProvider, IndexSnapshot as YahooIndexSnapshot
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False

try:
    from .alpha_vantage_ftse import AlphaVantageFTSEProvider, IndexSnapshot as AlphaVantageIndexSnapshot
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FTSEIndexSnapshot:
    """Unified index snapshot data."""
    timestamp: pd.Timestamp
    index_px: float
    volume: Optional[int] = None
    source: str = "unknown"


@dataclass
class FTSEOptionsSnapshot:
    """Unified options snapshot data."""
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
    source: str = "unknown"


class FTSEDataManager:
    """
    Unified FTSE 100 data manager with automatic fallback.
    
    Tries Intrinio first (if available and working), then falls back to Yahoo Finance.
    """
    
    def __init__(self, preferred_provider: str = "auto"):
        """
        Initialize FTSE data manager.
        
        Args:
            preferred_provider: "intrinio", "yahoo", or "auto"
        """
        self.preferred_provider = preferred_provider
        self.providers = {}
        self.working_provider = None
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available data providers."""
        # Try Alpha Vantage first (most reliable for FTSE 100)
        if ALPHA_VANTAGE_AVAILABLE and self.preferred_provider in ["alpha_vantage", "auto"]:
            try:
                alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
                if alpha_vantage_key:
                    self.providers['alpha_vantage'] = AlphaVantageFTSEProvider(alpha_vantage_key)
                    logger.info("Alpha Vantage provider initialized")
                else:
                    logger.warning("Alpha Vantage API key not found")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpha Vantage provider: {e}")
        
        # Try Intrinio second
        if INTRINIO_AVAILABLE and self.preferred_provider in ["intrinio", "auto"]:
            try:
                intrinio_key = os.getenv('INTRINIO_API_KEY')
                if intrinio_key:
                    self.providers['intrinio'] = IntrinioFTSEProvider(intrinio_key)
                    logger.info("Intrinio provider initialized")
                else:
                    logger.warning("Intrinio API key not found")
            except Exception as e:
                logger.warning(f"Failed to initialize Intrinio provider: {e}")
        
        # Initialize Yahoo Finance as fallback
        if YAHOO_AVAILABLE and self.preferred_provider in ["yahoo", "auto"]:
            try:
                self.providers['yahoo'] = YahooFTSEProvider()
                logger.info("Yahoo Finance provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Yahoo provider: {e}")
        
        if not self.providers:
            raise RuntimeError("No data providers available")
    
    def _test_provider(self, provider_name: str) -> bool:
        """Test if a provider is working."""
        try:
            provider = self.providers[provider_name]
            return provider.test_connection()
        except Exception as e:
            logger.warning(f"Provider {provider_name} test failed: {e}")
            return False
    
    def _get_working_provider(self) -> str:
        """Get the first working provider."""
        if self.working_provider and self._test_provider(self.working_provider):
            return self.working_provider
        
        # Try providers in order of preference
        provider_order = []
        if self.preferred_provider == "alpha_vantage":
            provider_order = ["alpha_vantage", "intrinio", "yahoo"]
        elif self.preferred_provider == "intrinio":
            provider_order = ["intrinio", "alpha_vantage", "yahoo"]
        elif self.preferred_provider == "yahoo":
            provider_order = ["yahoo", "alpha_vantage", "intrinio"]
        else:  # auto
            provider_order = ["alpha_vantage", "intrinio", "yahoo"]
        
        for provider_name in provider_order:
            if provider_name in self.providers and self._test_provider(provider_name):
                self.working_provider = provider_name
                logger.info(f"Using {provider_name} provider")
                return provider_name
        
        raise RuntimeError("No working data providers available")
    
    def get_index_snapshot(self, 
                          date: str, 
                          near_time: str = "15:45", 
                          window: int = 20) -> FTSEIndexSnapshot:
        """
        Get FTSE 100 index snapshot.
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format
            window: Window in minutes around target time
            
        Returns:
            FTSEIndexSnapshot with unified data format
        """
        provider_name = self._get_working_provider()
        provider = self.providers[provider_name]
        
        try:
            snapshot = provider.get_index_snapshot(date, near_time, window)
            
            # Convert to unified format
            return FTSEIndexSnapshot(
                timestamp=snapshot.timestamp,
                index_px=snapshot.index_px,
                volume=snapshot.volume,
                source=provider_name
            )
            
        except Exception as e:
            logger.error(f"Failed to get index snapshot from {provider_name}: {e}")
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
        
        Args:
            date: Date in YYYY-MM-DD format
            near_time: Target time in HH:MM format
            window: Window in minutes around target time
            maturity_bounds: (min_days, max_days) for option maturity
            spread_limit: Maximum relative spread
            min_oi: Minimum open interest
            
        Returns:
            DataFrame with options data
        """
        provider_name = self._get_working_provider()
        provider = self.providers[provider_name]
        
        try:
            options_df = provider.get_option_chain_snapshot(
                date, near_time, window, maturity_bounds, spread_limit, min_oi
            )
            
            # Add source column
            if not options_df.empty:
                options_df['source'] = provider_name
            
            return options_df
            
        except Exception as e:
            logger.error(f"Failed to get options data from {provider_name}: {e}")
            raise
    
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
        provider_name = self._get_working_provider()
        provider = self.providers[provider_name]
        
        try:
            data = provider.get_historical_data(start_date, end_date)
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical data from {provider_name}: {e}")
            raise
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_working_provider(self) -> Optional[str]:
        """Get currently working provider."""
        try:
            return self._get_working_provider()
        except RuntimeError:
            return None


# Convenience functions
def get_ftse_index_snapshot(date: str, 
                           near_time: str = "15:45", 
                           window: int = 20,
                           provider: str = "auto") -> FTSEIndexSnapshot:
    """
    Get FTSE 100 index snapshot with automatic provider selection.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        provider: Preferred provider ("intrinio", "yahoo", "auto")
        
    Returns:
        FTSEIndexSnapshot with unified data format
    """
    manager = FTSEDataManager(provider)
    return manager.get_index_snapshot(date, near_time, window)


def get_ftse_options_snapshot(date: str,
                             near_time: str = "15:45",
                             window: int = 20,
                             maturity_bounds: Tuple[int, int] = (7, 60),
                             spread_limit: float = 0.5,
                             min_oi: int = 100,
                             provider: str = "auto") -> pd.DataFrame:
    """
    Get FTSE 100 options chain snapshot with automatic provider selection.
    
    Args:
        date: Date in YYYY-MM-DD format
        near_time: Target time in HH:MM format
        window: Window in minutes around target time
        maturity_bounds: (min_days, max_days) for option maturity
        spread_limit: Maximum relative spread
        min_oi: Minimum open interest
        provider: Preferred provider ("intrinio", "yahoo", "auto")
        
    Returns:
        DataFrame with options data
    """
    manager = FTSEDataManager(provider)
    return manager.get_option_chain_snapshot(
        date, near_time, window, maturity_bounds, spread_limit, min_oi
    )


def get_ftse_historical_data(start_date: str, 
                            end_date: str,
                            provider: str = "auto") -> pd.DataFrame:
    """
    Get historical FTSE 100 data with automatic provider selection.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        provider: Preferred provider ("intrinio", "yahoo", "auto")
        
    Returns:
        DataFrame with OHLCV data
    """
    manager = FTSEDataManager(provider)
    return manager.get_historical_data(start_date, end_date)
