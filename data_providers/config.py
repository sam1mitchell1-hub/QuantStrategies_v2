"""
Configuration management for data providers.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class DataProviderConfig:
    """Configuration for data providers."""
    
    # Intrinio API
    intrinio_api_key: Optional[str] = None
    
    # Alpha Vantage API
    alpha_vantage_api_key: Optional[str] = None
    
    # Data provider settings
    default_provider: str = "intrinio"
    ftse_symbol: str = "UKX"
    default_timezone: str = "Europe/London"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'DataProviderConfig':
        """Load configuration from environment variables."""
        return cls(
            intrinio_api_key=os.getenv('INTRINIO_API_KEY'),
            alpha_vantage_api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            default_provider=os.getenv('DEFAULT_DATA_PROVIDER', 'intrinio'),
            ftse_symbol=os.getenv('FTSE_SYMBOL', 'UKX'),
            default_timezone=os.getenv('DEFAULT_TIMEZONE', 'Europe/London'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE')
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.default_provider == 'intrinio' and not self.intrinio_api_key:
            raise ValueError("INTRINIO_API_KEY is required when using intrinio provider")
        
        if self.default_provider == 'alpha_vantage' and not self.alpha_vantage_api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required when using alpha_vantage provider")
