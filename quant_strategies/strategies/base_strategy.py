from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..execution.models import Signal

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
        self.signals = []    # Track all signals generated (for backtesting)
        # Note: positions are now managed by Portfolio class in live trading
        
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
    
    def emit_signal(self, ticker: str, signal_type: SignalType, 
                   strength: float = 1.0) -> 'Signal':
        """
        Emit a trading signal for OMS processing.
        
        This method creates a Signal object that can be sent to the
        Order Management System for execution.
        
        Args:
            ticker: Stock ticker symbol
            signal_type: BUY, SELL, or HOLD
            strength: Signal strength/confidence (0.0 to 1.0)
            
        Returns:
            Signal object ready for OMS processing
        """
        from ..execution.models import Signal as OMSSignal, SignalType as OMSSignalType
        
        # Convert strategy SignalType to OMS SignalType
        if signal_type == SignalType.BUY:
            oms_signal_type = OMSSignalType.BUY
        elif signal_type == SignalType.SELL:
            oms_signal_type = OMSSignalType.SELL
        else:
            # HOLD signals are not actionable
            raise ValueError(f"Cannot emit HOLD signal - only BUY or SELL allowed")
        
        # Create OMS signal
        signal = OMSSignal.create(
            strategy_name=self.name,
            ticker=ticker,
            signal_type=oms_signal_type,
            strength=strength
        )
        
        # Track for backtesting purposes
        self.signals.append(signal)
        
        return signal