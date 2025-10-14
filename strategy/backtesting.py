"""
Backtesting Framework

Comprehensive backtesting system for the options trading strategy:
- Event-aligned data processing
- Realistic execution simulation
- Performance metrics and reporting
- Risk-adjusted returns analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    timestamp: pd.Timestamp
    underlying: str
    structure_type: str
    long_strike: float
    short_strike: float
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: Optional[str] = None
    expected_value: float = 0.0
    forecast_sharpe: float = 0.0


@dataclass
class BacktestResults:
    """Backtesting results container."""
    trades: List[Trade]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    trade_analysis: Dict[str, Any]
    monthly_returns: pd.Series
    drawdown_series: pd.Series
    config: Any


class Backtester:
    """Backtesting engine for options trading strategy."""
    
    def __init__(self, config):
        """Initialize backtester with configuration."""
        self.config = config
        self.trades = []
        self.current_positions = {}
        self.portfolio_value = config.risk.max_position_size_pct * 100000  # Starting value
        self.cash = self.portfolio_value
        self.portfolio_history = []
        
    def run_backtest(self, 
                    feature_data: pd.DataFrame,
                    market_data: Dict[str, pd.DataFrame],
                    model,
                    start_date: Optional[pd.Timestamp] = None,
                    end_date: Optional[pd.Timestamp] = None) -> BacktestResults:
        """
        Run complete backtest.
        
        Args:
            feature_data: Historical feature data
            market_data: Historical market data (prices, options, etc.)
            model: Trained forecasting model
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResults with comprehensive analysis
        """
        print("Starting backtest...")
        
        # Filter data by date range
        if start_date:
            feature_data = feature_data[feature_data.index >= start_date]
        if end_date:
            feature_data = feature_data[feature_data.index <= end_date]
        
        # Get trading schedule
        trading_schedule = self.config.get_trading_schedule(
            feature_data.index[0].date(),
            feature_data.index[-1].date()
        )
        
        # Initialize components
        from .features import FeatureBuilder
        from .forecasting import GBTForecaster
        from .structures import OptionStructureSelector
        from .pricing import FDPricer
        from .risk import RiskManager
        
        feature_builder = FeatureBuilder(self.config)
        forecaster = GBTForecaster(self.config)
        structure_selector = OptionStructureSelector(self.config)
        pricer = FDPricer(self.config)
        risk_manager = RiskManager(self.config)
        
        # Set up forecaster with trained model
        forecaster.model = model
        forecaster.is_trained = True
        
        # Run backtest
        for _, schedule_row in trading_schedule.iterrows():
            decision_time = schedule_row['decision_time']
            horizon_end = schedule_row['horizon_end']
            
            # Get data for this decision time
            decision_data = self._get_decision_data(
                decision_time, feature_data, market_data
            )
            
            if decision_data is None:
                continue
            
            # Make trading decision
            self._process_decision(
                decision_time, decision_data, forecaster, structure_selector, 
                pricer, risk_manager
            )
            
            # Update portfolio value
            self._update_portfolio_value(decision_time, market_data)
            
            # Check for position exits
            self._check_position_exits(decision_time, horizon_end, market_data)
        
        # Close remaining positions
        self._close_all_positions(feature_data.index[-1], market_data)
        
        # Calculate results
        results = self._calculate_results()
        
        print(f"Backtest completed. {len(self.trades)} trades executed.")
        return results
    
    def _get_decision_data(self, 
                          decision_time: pd.Timestamp,
                          feature_data: pd.DataFrame,
                          market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Get data for a specific decision time."""
        # Get feature data
        feature_row = feature_data.loc[feature_data.index <= decision_time].iloc[-1]
        
        # Get market data
        price_data = market_data.get('prices', pd.DataFrame())
        options_data = market_data.get('options', pd.DataFrame())
        
        if price_data.empty:
            return None
        
        # Get current price
        current_price = price_data.loc[price_data.index <= decision_time, 'close'].iloc[-1]
        
        # Get options data for this time
        if not options_data.empty:
            current_options = options_data.loc[options_data.index <= decision_time]
            if not current_options.empty:
                current_options = current_options.iloc[-1]
            else:
                current_options = pd.Series()
        else:
            current_options = pd.Series()
        
        return {
            'features': feature_row,
            'current_price': current_price,
            'options': current_options,
            'timestamp': decision_time
        }
    
    def _process_decision(self, 
                         decision_time: pd.Timestamp,
                         decision_data: Dict[str, Any],
                         forecaster,
                         structure_selector,
                         pricer,
                         risk_manager):
        """Process trading decision for a specific time."""
        # Get forecast
        features = decision_data['features'].to_frame().T
        forecast_results = forecaster.predict(features)
        
        # Convert to dictionary format
        forecast = {
            'expected_returns': 0.0,  # Default
            'probabilities': 0.5      # Default
        }
        
        if forecast_results.probabilities is not None:
            forecast['probabilities'] = forecast_results.probabilities[0]
            # Convert probability to expected return
            confidence = abs(forecast_results.probabilities[0] - 0.5) * 2
            forecast['expected_returns'] = (forecast_results.probabilities[0] - 0.5) * confidence * 0.02
        
        # Check if we should trade
        if abs(forecast.get('expected_returns', 0)) < 0.001:  # Minimum threshold
            return
        
        # Select structure
        market_data = {
            'options': [decision_data['options'].to_dict()] if not decision_data['options'].empty else []
        }
        
        structure_selection = structure_selector.select_structure(
            forecast, market_data, decision_data['current_price'], decision_time
        )
        
        if structure_selection.structure is None:
            return
        
        # Price the structure
        structure_pricing = pricer.price_structure(
            structure_selection.structure,
            decision_data['current_price'],
            0.05,  # Risk-free rate
            0.0,   # Dividend yield
            0.2    # Implied volatility
        )
        
        # Calculate expected value
        expected_value = pricer.calculate_expected_value(
            structure_selection.structure,
            forecast,
            decision_data['current_price'],
            0.05,
            0.0
        )
        
        # Check trade eligibility
        is_eligible, reason = risk_manager.check_trade_eligibility(
            structure_selection.structure,
            expected_value,
            decision_data['current_price']
        )
        
        if not is_eligible:
            return
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            expected_value,
            structure_selection.structure.max_loss,
            decision_data['current_price']
        )
        
        if position_size <= 0:
            return
        
        # Execute trade
        self._execute_trade(
            decision_time, structure_selection, structure_pricing,
            expected_value, position_size, decision_data['current_price']
        )
    
    def _execute_trade(self, 
                      timestamp: pd.Timestamp,
                      structure_selection,
                      structure_pricing,
                      expected_value: float,
                      quantity: int,
                      current_price: float):
        """Execute a trade."""
        # Calculate costs
        commission = quantity * self.config.execution.commission_per_contract * 2  # Both legs
        exchange_fees = quantity * self.config.execution.exchange_fees_per_contract * 2
        slippage = quantity * current_price * self.config.execution.slippage_bps / 10000
        
        total_costs = commission + exchange_fees + slippage
        
        # Create trade record
        trade = Trade(
            trade_id=f"trade_{len(self.trades) + 1}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            underlying="SPY",  # Default underlying
            structure_type=structure_selection.structure.structure_type.value,
            long_strike=structure_selection.structure.long_leg.strike_price,
            short_strike=structure_selection.structure.short_leg.strike_price,
            quantity=quantity,
            entry_price=structure_pricing.net_price,
            commission=commission,
            slippage=slippage,
            expected_value=expected_value,
            forecast_sharpe=structure_selection.forecast_sharpe
        )
        
        # Add to trades
        self.trades.append(trade)
        
        # Add to current positions
        self.current_positions[trade.trade_id] = trade
        
        # Update cash
        self.cash -= (structure_pricing.net_price * quantity + total_costs)
        
        print(f"Executed trade: {trade.trade_id}, Quantity: {quantity}, Price: {structure_pricing.net_price:.4f}")
    
    def _check_position_exits(self, 
                             current_time: pd.Timestamp,
                             horizon_end: pd.Timestamp,
                             market_data: Dict[str, pd.DataFrame]):
        """Check for position exits."""
        positions_to_close = []
        
        for trade_id, trade in self.current_positions.items():
            exit_reason = None
            
            # Time-based exit
            if current_time >= horizon_end:
                exit_reason = "time_horizon"
            
            # Check for model flip (simplified)
            # In practice, you'd re-run the model and check if forecast changed
            
            # Check for risk limits (simplified)
            # In practice, you'd check current Greeks and risk metrics
            
            if exit_reason:
                positions_to_close.append((trade_id, exit_reason))
        
        # Close positions
        for trade_id, exit_reason in positions_to_close:
            self._close_position(trade_id, current_time, exit_reason, market_data)
    
    def _close_position(self, 
                       trade_id: str,
                       exit_time: pd.Timestamp,
                       exit_reason: str,
                       market_data: Dict[str, pd.DataFrame]):
        """Close a position."""
        if trade_id not in self.current_positions:
            return
        
        trade = self.current_positions[trade_id]
        
        # Calculate exit price (simplified - in practice, you'd use current market prices)
        # For now, assume we can close at the same price
        exit_price = trade.entry_price
        
        # Calculate P&L
        pnl = (exit_price - trade.entry_price) * trade.quantity
        
        # Calculate costs
        commission = trade.quantity * self.config.execution.commission_per_contract * 2
        exchange_fees = trade.quantity * self.config.execution.exchange_fees_per_contract * 2
        slippage = trade.quantity * exit_price * self.config.execution.slippage_bps / 10000
        
        total_costs = commission + exchange_fees + slippage
        net_pnl = pnl - total_costs
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_timestamp = exit_time
        trade.pnl = net_pnl
        trade.exit_reason = exit_reason
        trade.commission += commission
        trade.slippage += slippage
        
        # Update cash
        self.cash += (exit_price * trade.quantity - total_costs)
        
        # Remove from current positions
        del self.current_positions[trade_id]
        
        print(f"Closed position: {trade_id}, P&L: {net_pnl:.2f}, Reason: {exit_reason}")
    
    def _close_all_positions(self, end_time: pd.Timestamp, market_data: Dict[str, pd.DataFrame]):
        """Close all remaining positions."""
        for trade_id in list(self.current_positions.keys()):
            self._close_position(trade_id, end_time, "end_of_backtest", market_data)
    
    def _update_portfolio_value(self, timestamp: pd.Timestamp, market_data: Dict[str, pd.DataFrame]):
        """Update portfolio value."""
        # Calculate unrealized P&L for current positions
        unrealized_pnl = 0.0
        for trade in self.current_positions.values():
            # Simplified - assume no change in value
            unrealized_pnl += 0.0
        
        # Total portfolio value
        total_value = self.cash + unrealized_pnl
        
        # Store in history
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'total_value': total_value
        })
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        if not self.trades:
            return BacktestResults(
                trades=[],
                performance_metrics={},
                risk_metrics={},
                trade_analysis={},
                monthly_returns=pd.Series(),
                drawdown_series=pd.Series(),
                config=self.config
            )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # Analyze trades
        trade_analysis = self._analyze_trades()
        
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns()
        
        # Calculate drawdown series
        drawdown_series = self._calculate_drawdown_series()
        
        return BacktestResults(
            trades=self.trades,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            trade_analysis=trade_analysis,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown_series,
            config=self.config
        )
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        
        # Calculate returns
        if self.portfolio_history:
            initial_value = self.portfolio_history[0]['total_value']
            final_value = self.portfolio_history[-1]['total_value']
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.trades) > 1:
            pnl_values = [t.pnl for t in self.trades]
            sharpe_ratio = np.mean(pnl_values) / np.std(pnl_values) if np.std(pnl_values) > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        # Calculate hit rate
        hit_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'hit_rate': hit_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'net_pnl': total_pnl - total_commission - total_slippage
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.trades:
            return {}
        
        # Calculate VaR
        pnl_values = [t.pnl for t in self.trades]
        var_95 = np.percentile(pnl_values, 5) if pnl_values else 0
        var_99 = np.percentile(pnl_values, 1) if pnl_values else 0
        
        # Calculate maximum drawdown
        if self.portfolio_history:
            values = [h['total_value'] for h in self.portfolio_history]
            peak = values[0]
            max_dd = 0
            for value in values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0
        
        # Calculate volatility
        if len(pnl_values) > 1:
            volatility = np.std(pnl_values)
        else:
            volatility = 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_dd,
            'volatility': volatility
        }
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade characteristics."""
        if not self.trades:
            return {}
        
        # Analyze by structure type
        structure_analysis = {}
        for structure_type in set(t.structure_type for t in self.trades):
            structure_trades = [t for t in self.trades if t.structure_type == structure_type]
            if structure_trades:
                structure_analysis[structure_type] = {
                    'count': len(structure_trades),
                    'hit_rate': len([t for t in structure_trades if t.pnl > 0]) / len(structure_trades),
                    'avg_pnl': np.mean([t.pnl for t in structure_trades]),
                    'total_pnl': sum(t.pnl for t in structure_trades)
                }
        
        # Analyze by exit reason
        exit_analysis = {}
        for exit_reason in set(t.exit_reason for t in self.trades if t.exit_reason):
            exit_trades = [t for t in self.trades if t.exit_reason == exit_reason]
            if exit_trades:
                exit_analysis[exit_reason] = {
                    'count': len(exit_trades),
                    'hit_rate': len([t for t in exit_trades if t.pnl > 0]) / len(exit_trades),
                    'avg_pnl': np.mean([t.pnl for t in exit_trades]),
                    'total_pnl': sum(t.pnl for t in exit_trades)
                }
        
        return {
            'structure_analysis': structure_analysis,
            'exit_analysis': exit_analysis,
            'total_trades': len(self.trades)
        }
    
    def _calculate_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        if not self.portfolio_history:
            return pd.Series()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to monthly
        monthly_values = df['total_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        return monthly_returns
    
    def _calculate_drawdown_series(self) -> pd.Series:
        """Calculate drawdown series."""
        if not self.portfolio_history:
            return pd.Series()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate running maximum
        df['peak'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['peak']) / df['peak']
        
        return df['drawdown']
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'underlying': trade.underlying,
                'structure_type': trade.structure_type,
                'long_strike': trade.long_strike,
                'short_strike': trade.short_strike,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'exit_timestamp': trade.exit_timestamp,
                'pnl': trade.pnl,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'exit_reason': trade.exit_reason,
                'expected_value': trade.expected_value,
                'forecast_sharpe': trade.forecast_sharpe
            })
        
        return pd.DataFrame(data)
