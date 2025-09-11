import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy, SignalType

class BollingerRSIStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.
    
    Strategy Logic:
    - Buy when price touches lower Bollinger Band AND RSI < 30 (oversold)
    - Sell when price touches upper Bollinger Band AND RSI > 70 (overbought)
    - Add volume confirmation for stronger signals
    """
    
    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize the Bollinger Bands + RSI strategy.
        
        Args:
            parameters: Strategy parameters
                - bb_period: Bollinger Bands period (default: 20)
                - bb_std: Bollinger Bands standard deviation (default: 2)
                - rsi_period: RSI period (default: 14)
                - rsi_oversold: RSI oversold threshold (default: 30)
                - rsi_overbought: RSI overbought threshold (default: 70)
                - volume_multiplier: Volume confirmation multiplier (default: 1.5)
        """
        default_params = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_multiplier': 1.5
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Bollinger_RSI_Mean_Reversion", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and RSI indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator columns added
        """
        df = data.copy()
        
        # Calculate Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        
        # Calculate moving average
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std'])
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands and RSI.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with signal columns added
        """
        df = data.copy()
        
        # Initialize signal column
        df['signal'] = SignalType.HOLD
        
        # Get parameters
        rsi_oversold = self.parameters['rsi_oversold']
        rsi_overbought = self.parameters['rsi_overbought']
        volume_multiplier = self.parameters['volume_multiplier']
        
        # Generate buy signals
        # Buy when price touches lower Bollinger Band AND RSI is oversold
        buy_condition = (
            (df['close'] <= df['bb_lower']) & 
            (df['rsi'] < rsi_oversold) &
            (df['volume_ratio'] > volume_multiplier)  # Volume confirmation
        )
        df.loc[buy_condition, 'signal'] = SignalType.BUY
        
        # Generate sell signals
        # Sell when price touches upper Bollinger Band AND RSI is overbought
        sell_condition = (
            (df['close'] >= df['bb_upper']) & 
            (df['rsi'] > rsi_overbought) &
            (df['volume_ratio'] > volume_multiplier)  # Volume confirmation
        )
        df.loc[sell_condition, 'signal'] = SignalType.SELL
        
        # Add signal strength (how far from the band)
        df['signal_strength'] = 0.0
        
        # Calculate signal strength for buy signals
        buy_signals = df['signal'] == SignalType.BUY
        if buy_signals.any():
            df.loc[buy_signals, 'signal_strength'] = (
                (df.loc[buy_signals, 'bb_lower'] - df.loc[buy_signals, 'close']) / 
                df.loc[buy_signals, 'close']
            )
        
        # Calculate signal strength for sell signals
        sell_signals = df['signal'] == SignalType.SELL
        if sell_signals.any():
            df.loc[sell_signals, 'signal_strength'] = (
                (df.loc[sell_signals, 'close'] - df.loc[sell_signals, 'bb_upper']) / 
                df.loc[sell_signals, 'close']
            )
        
        return df
    
    def get_required_columns(self) -> list:
        """Return required columns for this strategy."""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_indicator_columns(self) -> list:
        """Return indicator columns added by this strategy."""
        return [
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
            'rsi', 'volume_ma', 'volume_ratio',
            'signal', 'signal_strength'
        ] 