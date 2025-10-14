"""
Live Strategy Runner

Bridges DataFrame-based strategies with the OMS for live trading.
Handles data fetching, signal generation, and order execution.
"""

from typing import List, Dict, Optional, TYPE_CHECKING
import logging
import pandas as pd

from .base_strategy import BaseStrategy, SignalType

if TYPE_CHECKING:
    from ..execution.oms import OrderManagementSystem
    from ..data.live_data_fetcher import LiveDataFetcher

logger = logging.getLogger(__name__)


class LiveStrategyRunner:
    """
    Runs a strategy in live mode with the OMS.
    
    Handles:
    - Fetching historical data for indicators
    - Running strategy calculations
    - Converting DataFrame signals to OMS signals
    - Executing trades through OMS
    """
    
    def __init__(self, 
                 strategy: BaseStrategy,
                 oms: 'OrderManagementSystem',
                 data_fetcher: 'LiveDataFetcher',
                 watchlist: List[str],
                 bars_for_indicators: int = 100):
        """
        Initialize live strategy runner.
        
        Args:
            strategy: Strategy instance to run
            oms: Order Management System
            data_fetcher: Live data fetcher
            watchlist: List of tickers to monitor
            bars_for_indicators: Number of historical bars for indicators
        """
        self.strategy = strategy
        self.oms = oms
        self.fetcher = data_fetcher
        self.watchlist = watchlist
        self.bars_for_indicators = bars_for_indicators
        
        # Track last signals to avoid duplicate trades
        self.last_signals: Dict[str, SignalType] = {}
        
        logger.info(f"LiveStrategyRunner initialized: {strategy.name} monitoring {len(watchlist)} tickers")
    
    def run_cycle(self) -> Dict[str, any]:
        """
        Run one trading cycle for all tickers in watchlist.
        
        Returns:
            Dictionary with cycle results
        """
        logger.info(f"Running trading cycle for {len(self.watchlist)} tickers")
        
        cycle_results = {
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_filled': 0,
            'errors': 0,
            'tickers_processed': []
        }
        
        for ticker in self.watchlist:
            try:
                result = self._process_ticker(ticker)
                cycle_results['tickers_processed'].append({
                    'ticker': ticker,
                    'signal': result.get('signal'),
                    'order_placed': result.get('order_placed', False)
                })
                
                if result.get('signal') != SignalType.HOLD:
                    cycle_results['signals_generated'] += 1
                
                if result.get('order_placed'):
                    cycle_results['orders_placed'] += 1
                
                if result.get('order_filled'):
                    cycle_results['orders_filled'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)
                cycle_results['errors'] += 1
        
        logger.info(f"Cycle complete: {cycle_results['signals_generated']} signals, "
                   f"{cycle_results['orders_placed']} orders placed")
        
        return cycle_results
    
    def _process_ticker(self, ticker: str) -> Dict:
        """
        Process a single ticker: fetch data, generate signals, execute trades.
        
        Args:
            ticker: Stock ticker to process
            
        Returns:
            Dictionary with processing results
        """
        logger.debug(f"Processing {ticker}")
        
        # Get historical data for indicators
        hist_data = self.fetcher.get_latest_bars(
            ticker, 
            n=self.bars_for_indicators,
            interval='1d'
        )
        
        if hist_data is None or hist_data.empty:
            logger.warning(f"No historical data for {ticker}, skipping")
            return {'signal': SignalType.HOLD, 'order_placed': False}
        
        # Validate data has required columns
        if not self.strategy.validate_data(hist_data):
            logger.warning(f"Invalid data for {ticker}, skipping")
            return {'signal': SignalType.HOLD, 'order_placed': False}
        
        # Calculate indicators
        try:
            data_with_indicators = self.strategy.calculate_indicators(hist_data)
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            return {'signal': SignalType.HOLD, 'order_placed': False}
        
        # Generate signals
        try:
            data_with_signals = self.strategy.generate_signals(data_with_indicators)
        except Exception as e:
            logger.error(f"Error generating signals for {ticker}: {e}")
            return {'signal': SignalType.HOLD, 'order_placed': False}
        
        # Get latest signal
        latest_signal = data_with_signals.iloc[-1]['signal']
        
        # Convert if it's an enum value
        if hasattr(latest_signal, 'value'):
            signal_str = latest_signal.value
        else:
            signal_str = str(latest_signal)
        
        # Map to SignalType enum
        if signal_str == 'BUY' or signal_str == SignalType.BUY.value:
            signal_type = SignalType.BUY
        elif signal_str == 'SELL' or signal_str == SignalType.SELL.value:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        logger.debug(f"{ticker}: Signal = {signal_type.value}")
        
        # Check if signal changed (avoid duplicate orders)
        last_signal = self.last_signals.get(ticker, SignalType.HOLD)
        if signal_type == last_signal and signal_type != SignalType.HOLD:
            logger.debug(f"{ticker}: Signal unchanged ({signal_type.value}), skipping")
            return {'signal': signal_type, 'order_placed': False}
        
        # Update last signal
        self.last_signals[ticker] = signal_type
        
        # Process signal through OMS if actionable
        if signal_type != SignalType.HOLD:
            return self._execute_signal(ticker, signal_type, data_with_signals)
        
        return {'signal': SignalType.HOLD, 'order_placed': False}
    
    def _execute_signal(self, ticker: str, signal_type: SignalType, 
                       data: pd.DataFrame) -> Dict:
        """
        Execute a trading signal through the OMS.
        
        Args:
            ticker: Stock ticker
            signal_type: BUY or SELL
            data: DataFrame with signals and indicators
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Get signal strength if available
            if 'signal_strength' in data.columns:
                strength = abs(float(data.iloc[-1]['signal_strength']))
                strength = min(max(strength, 0.0), 1.0)  # Clamp to [0, 1]
            else:
                strength = 0.75  # Default strength
            
            # Emit OMS signal
            oms_signal = self.strategy.emit_signal(ticker, signal_type, strength)
            logger.info(f"Emitted signal: {oms_signal.signal_id} - {ticker} {signal_type.value}")
            
            # Get current market data
            market_data = self.fetcher.get_market_data(ticker)
            
            if market_data is None:
                logger.error(f"No market data for {ticker}, cannot execute")
                return {'signal': signal_type, 'order_placed': False}
            
            # Process through OMS
            order = self.oms.process_signal(oms_signal, market_data)
            
            if order is None:
                logger.info(f"Order rejected for {ticker}")
                return {'signal': signal_type, 'order_placed': False, 'order_filled': False}
            
            # Check if order was filled
            from ..execution.models import OrderStatus
            order_filled = (order.status == OrderStatus.FILLED)
            
            logger.info(f"Order processed: {order.order_id} - Status: {order.status.value}")
            
            return {
                'signal': signal_type,
                'order_placed': True,
                'order_filled': order_filled,
                'order': order
            }
            
        except Exception as e:
            logger.error(f"Error executing signal for {ticker}: {e}", exc_info=True)
            return {'signal': signal_type, 'order_placed': False, 'error': str(e)}
    
    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status.
        
        Returns:
            Portfolio summary dictionary
        """
        # Get current prices for all positions
        tickers = list(self.oms.portfolio.positions.keys())
        current_prices = {}
        
        if tickers:
            current_prices = self.fetcher.get_current_prices(tickers)
        
        return self.oms.get_portfolio_status(current_prices)
    
    def print_portfolio_summary(self):
        """Print formatted portfolio summary."""
        # Get current prices
        tickers = list(self.oms.portfolio.positions.keys())
        current_prices = {}
        
        if tickers:
            current_prices = self.fetcher.get_current_prices(tickers)
        
        self.oms.print_portfolio_summary(current_prices)
    
    def reset_signal_memory(self):
        """Clear signal memory (useful for testing)."""
        self.last_signals.clear()
        logger.info("Signal memory reset")

