"""
Saxo Bank API data provider for fetching historical equity data.
"""
import json
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SaxoDataProvider:
    """
    Data provider for Saxo Bank API to fetch historical equity data.
    """
    
    def __init__(self, config_path: str = "data_providers/saxo_config.json"):
        """
        Initialize the Saxo data provider.
        
        Args:
            config_path: Path to the configuration file containing API key
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config["api_key"]}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            if config.get("api_key") == "YOUR_24HR_API_KEY_HERE":
                logger.warning("Please update your API key in saxo_config.json")
            
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Make a request to the Saxo API with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response data or None if failed
        """
        url = f"{self.config['base_url']}{endpoint}"
        
        for attempt in range(self.config.get('retry_attempts', 3)):
            try:
                response = self.session.get(url, params=params, timeout=self.config.get('timeout', 30))
                response.raise_for_status()
                
                data = response.json()
                if 'Data' in data:
                    return data['Data']
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.get('retry_attempts', 3) - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All retry attempts failed for endpoint: {endpoint}")
                    return None
    
    def search_instruments(self, query: str, asset_type: str = "Stock") -> List[Dict[str, Any]]:
        """
        Search for instruments by name or symbol.
        
        Args:
            query: Search query (e.g., "Microsoft", "AAPL")
            asset_type: Type of asset to search for
            
        Returns:
            List of matching instruments
        """
        endpoint = "/ref/v1/instruments"
        params = {
            'AssetTypes': asset_type,
            'Keywords': query,
            'IncludeNonTradable': False
        }
        
        data = self._make_request(endpoint, params)
        if data:
            logger.info(f"Found {len(data)} instruments for query: {query}")
            return data
        return []
    
    def get_instrument_details(self, uic: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific instrument.
        
        Args:
            uic: Universal Instrument Code
            
        Returns:
            Instrument details or None if not found
        """
        endpoint = f"/ref/v1/instruments/details/{uic}"
        return self._make_request(endpoint)
    
    def get_historical_data(self, 
                          uic: int, 
                          start_date: str, 
                          end_date: str,
                          interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for an instrument.
        
        Args:
            uic: Universal Instrument Code
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 5m, 1h, 1d, 1w, 1M)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Map interval to Saxo's horizon parameter
        horizon_map = {
            '1m': 1, '5m': 5, '1h': 60, '1d': 1440, '1w': 10080, '1M': 43200
        }
        
        if interval not in horizon_map:
            logger.error(f"Unsupported interval: {interval}")
            return None
        
        # Try different chart endpoints
        endpoint = "/chart/v1/charts"
        params = {
            'Uic': uic,
            'Horizon': horizon_map[interval],
            'From': start_date,
            'To': end_date,
            'Count': 1000  # Reduce count to avoid issues
        }
        
        data = self._make_request(endpoint, params)
        if not data:
            logger.error(f"No data received for UIC {uic}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning(f"Empty data received for UIC {uic}")
            return None
        
        # Rename columns to standard format
        column_mapping = {
            'Time': 'timestamp',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        logger.info(f"Retrieved {len(df)} data points for UIC {uic}")
        return df
    
    def get_equity_data(self, 
                       symbol: str, 
                       start_date: str, 
                       end_date: str,
                       interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical data for an equity by symbol.
        
        Args:
            symbol: Stock symbol (e.g., "MSFT", "TSLA")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Search for the instrument
        instruments = self.search_instruments(symbol, "Stock")
        if not instruments:
            logger.error(f"No instruments found for symbol: {symbol}")
            return None
        
        # Find the best match (exact symbol match preferred)
        best_match = None
        for instrument in instruments:
            if instrument.get('Symbol', '').upper() == symbol.upper():
                best_match = instrument
                break
        
        if not best_match:
            # Use the first result if no exact match
            best_match = instruments[0]
            logger.warning(f"No exact match for {symbol}, using: {best_match.get('Symbol', 'Unknown')}")
        
        # Try both Uic and Identifier fields
        uic = best_match.get('Uic') or best_match.get('Identifier')
        if not uic:
            logger.error(f"No UIC/Identifier found for instrument: {best_match}")
            return None
        
        logger.info(f"Using UIC {uic} for symbol {symbol}")
        return self.get_historical_data(uic, start_date, end_date, interval)
    
    def get_multiple_equities(self, 
                             symbols: List[str], 
                             start_date: str, 
                             end_date: str,
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple equities.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            try:
                data = self.get_equity_data(symbol, start_date, end_date, interval)
                if data is not None and not data.empty:
                    results[symbol] = data
                    logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test the API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get account information
            endpoint = "/port/v1/accounts/me"
            data = self._make_request(endpoint)
            if data:
                logger.info("Saxo API connection successful")
                return True
            else:
                logger.error("Saxo API connection failed")
                return False
        except Exception as e:
            logger.error(f"Saxo API connection test failed: {e}")
            return False
