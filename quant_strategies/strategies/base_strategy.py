from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common interface and utility methods.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.positions = {}  # Track current positions
        self.signals = []    # Track all signals generated
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with signal columns added
        """
        pass
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run a backtest of the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)
        
        # Generate signals
        data_with_signals = self.generate_signals(data_with_indicators)
        
        # Calculate performance metrics
        results = self._calculate_performance(data_with_signals)
        
        return results
    
    def _calculate_performance(self, data: pd.DataFrame) -> Dict:
        """
        Calculate basic performance metrics.
        
        Args:
            data: DataFrame with signals
            
        Returns:
            Dictionary with performance metrics
        """
        # This is a basic implementation - can be enhanced
        total_signals = len(data[data['signal'] != SignalType.HOLD])
        buy_signals = len(data[data['signal'] == SignalType.BUY])
        sell_signals = len(data[data['signal'] == SignalType.SELL])
        
        return {
            'strategy_name': self.name,
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'data_points': len(data)
        }
    
    def get_required_columns(self) -> List[str]:
        """
        Return list of required columns for this strategy.
        
        Returns:
            List of required column names
        """
        return ['open', 'high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        return True 