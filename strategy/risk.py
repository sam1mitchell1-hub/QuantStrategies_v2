"""
Risk Management

Comprehensive risk management for the trading strategy:
- Position sizing using Kelly criterion
- Risk limits and monitoring
- Portfolio-level risk aggregation
- Real-time risk alerts and controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Position information."""
    structure_id: str
    underlying: str
    structure_type: str
    quantity: int
    entry_price: float
    current_price: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    max_loss: float
    max_profit: float
    entry_time: pd.Timestamp
    expiry_time: pd.Timestamp
    expected_value: float
    unrealized_pnl: float = 0.0


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    total_exposure: float
    max_loss: float
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float


@dataclass
class RiskAlert:
    """Risk alert information."""
    alert_type: str
    level: RiskLevel
    message: str
    current_value: float
    limit_value: float
    timestamp: pd.Timestamp
    action_required: bool


class RiskManager:
    """Risk management system for options trading strategy."""
    
    def __init__(self, config):
        """Initialize risk manager with configuration."""
        self.config = config
        self.positions = {}  # Dictionary of active positions
        self.risk_history = []  # Historical risk metrics
        self.alerts = []  # Active risk alerts
        self.account_value = 100000.0  # Starting account value
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
    def add_position(self, position: Position) -> bool:
        """
        Add a new position to the portfolio.
        
        Args:
            position: Position to add
            
        Returns:
            True if position added successfully, False if rejected
        """
        # Check if position would violate risk limits
        if not self._check_position_limits(position):
            return False
        
        # Add position
        self.positions[position.structure_id] = position
        
        # Update risk metrics
        self._update_risk_metrics()
        
        # Check for new alerts
        self._check_risk_alerts()
        
        return True
    
    def remove_position(self, structure_id: str) -> bool:
        """
        Remove a position from the portfolio.
        
        Args:
            structure_id: ID of position to remove
            
        Returns:
            True if position removed successfully
        """
        if structure_id in self.positions:
            del self.positions[structure_id]
            self._update_risk_metrics()
            self._check_risk_alerts()
            return True
        return False
    
    def update_position(self, structure_id: str, **kwargs) -> bool:
        """
        Update position information.
        
        Args:
            structure_id: ID of position to update
            **kwargs: Fields to update
            
        Returns:
            True if position updated successfully
        """
        if structure_id not in self.positions:
            return False
        
        position = self.positions[structure_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)
        
        # Update risk metrics
        self._update_risk_metrics()
        
        # Check for new alerts
        self._check_risk_alerts()
        
        return True
    
    def calculate_position_size(self, 
                              expected_value: float,
                              max_loss: float,
                              current_price: float) -> int:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            expected_value: Expected value of the trade
            max_loss: Maximum loss per contract
            current_price: Current underlying price
            
        Returns:
            Number of contracts to trade
        """
        # Kelly fraction
        kelly_fraction = self.config.risk.kelly_fraction
        
        # Cap Kelly fraction
        kelly_fraction = min(kelly_fraction, self.config.risk.max_kelly_fraction)
        
        # Calculate Kelly position size
        if max_loss > 0:
            kelly_size = (expected_value / max_loss) * kelly_fraction
        else:
            kelly_size = 0
        
        # Apply position size limits
        max_position_value = self.account_value * self.config.risk.max_position_size_pct
        max_contracts = int(max_position_value / (max_loss * current_price))
        
        # Final position size
        position_size = min(int(kelly_size), max_contracts)
        
        # Ensure minimum position size
        if position_size < 1 and expected_value > 0:
            position_size = 1
        
        return position_size
    
    def check_trade_eligibility(self, 
                               structure,
                               expected_value: float,
                               current_price: float) -> Tuple[bool, str]:
        """
        Check if a trade is eligible given current risk limits.
        
        Args:
            structure: Option structure
            expected_value: Expected value of the trade
            current_price: Current underlying price
            
        Returns:
            Tuple of (is_eligible, reason)
        """
        # Check expected value threshold
        if expected_value < self.config.risk.min_expected_value:
            return False, f"Expected value {expected_value:.4f} below threshold {self.config.risk.min_expected_value:.4f}"
        
        # Check maximum loss per trade
        if structure.max_loss > self.account_value * self.config.risk.max_loss_per_trade:
            return False, f"Maximum loss {structure.max_loss:.2f} exceeds limit {self.account_value * self.config.risk.max_loss_per_trade:.2f}"
        
        # Check daily loss limit
        daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        if daily_pnl < -self.account_value * self.config.risk.max_daily_loss:
            return False, f"Daily loss {daily_pnl:.2f} exceeds limit {self.account_value * self.config.risk.max_daily_loss:.2f}"
        
        # Check drawdown limit
        if self.current_drawdown > self.config.risk.max_drawdown:
            return False, f"Current drawdown {self.current_drawdown:.2f} exceeds limit {self.config.risk.max_drawdown:.2f}"
        
        return True, "Trade eligible"
    
    def _check_position_limits(self, position: Position) -> bool:
        """Check if position violates risk limits."""
        # Check individual position limits
        if position.max_loss > self.account_value * self.config.risk.max_loss_per_trade:
            return False
        
        # Check portfolio-level limits (simplified)
        total_delta = sum(pos.net_delta for pos in self.positions.values()) + position.net_delta
        if abs(total_delta) > self.config.risk.max_delta_exposure:
            return False
        
        total_gamma = sum(pos.net_gamma for pos in self.positions.values()) + position.net_gamma
        if abs(total_gamma) > self.config.risk.max_gamma_exposure:
            return False
        
        total_theta = sum(pos.net_theta for pos in self.positions.values()) + position.net_theta
        if abs(total_theta) > self.config.risk.max_theta_exposure:
            return False
        
        total_vega = sum(pos.net_vega for pos in self.positions.values()) + position.net_vega
        if abs(total_vega) > self.config.risk.max_vega_exposure:
            return False
        
        return True
    
    def _update_risk_metrics(self):
        """Update portfolio risk metrics."""
        if not self.positions:
            # No positions - reset metrics
            self.risk_metrics = RiskMetrics(
                total_delta=0.0,
                total_gamma=0.0,
                total_theta=0.0,
                total_vega=0.0,
                total_rho=0.0,
                total_exposure=0.0,
                max_loss=0.0,
                var_95=0.0,
                var_99=0.0,
                expected_shortfall=0.0,
                sharpe_ratio=0.0,
                max_drawdown=self.max_drawdown,
                current_drawdown=self.current_drawdown
            )
            return
        
        # Calculate aggregate Greeks
        total_delta = sum(pos.net_delta for pos in self.positions.values())
        total_gamma = sum(pos.net_gamma for pos in self.positions.values())
        total_theta = sum(pos.net_theta for pos in self.positions.values())
        total_vega = sum(pos.net_vega for pos in self.positions.values())
        total_rho = sum(pos.net_rho for pos in self.positions.values())
        
        # Calculate total exposure
        total_exposure = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        
        # Calculate maximum loss
        max_loss = sum(pos.max_loss * pos.quantity for pos in self.positions.values())
        
        # Calculate VaR (simplified)
        pnl_values = [pos.unrealized_pnl for pos in self.positions.values()]
        if pnl_values:
            var_95 = np.percentile(pnl_values, 5)  # 95% VaR
            var_99 = np.percentile(pnl_values, 1)  # 99% VaR
            expected_shortfall = np.mean([p for p in pnl_values if p <= var_95])
        else:
            var_95 = var_99 = expected_shortfall = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if len(pnl_values) > 1:
            sharpe_ratio = np.mean(pnl_values) / np.std(pnl_values) if np.std(pnl_values) > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        # Update drawdown
        current_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        peak_value = self.account_value + max(0, current_pnl)
        self.current_drawdown = max(0, peak_value - (self.account_value + current_pnl)) / peak_value
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Create risk metrics
        self.risk_metrics = RiskMetrics(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            total_rho=total_rho,
            total_exposure=total_exposure,
            max_loss=max_loss,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            current_drawdown=self.current_drawdown
        )
        
        # Store in history
        self.risk_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': self.risk_metrics
        })
    
    def _check_risk_alerts(self):
        """Check for risk limit violations and create alerts."""
        if not hasattr(self, 'risk_metrics'):
            return
        
        # Clear old alerts
        self.alerts = []
        
        # Check delta exposure
        if abs(self.risk_metrics.total_delta) > self.config.risk.max_delta_exposure:
            self.alerts.append(RiskAlert(
                alert_type="delta_exposure",
                level=RiskLevel.HIGH,
                message=f"Delta exposure {self.risk_metrics.total_delta:.3f} exceeds limit {self.config.risk.max_delta_exposure:.3f}",
                current_value=self.risk_metrics.total_delta,
                limit_value=self.config.risk.max_delta_exposure,
                timestamp=pd.Timestamp.now(),
                action_required=True
            ))
        
        # Check gamma exposure
        if abs(self.risk_metrics.total_gamma) > self.config.risk.max_gamma_exposure:
            self.alerts.append(RiskAlert(
                alert_type="gamma_exposure",
                level=RiskLevel.HIGH,
                message=f"Gamma exposure {self.risk_metrics.total_gamma:.3f} exceeds limit {self.config.risk.max_gamma_exposure:.3f}",
                current_value=self.risk_metrics.total_gamma,
                limit_value=self.config.risk.max_gamma_exposure,
                timestamp=pd.Timestamp.now(),
                action_required=True
            ))
        
        # Check theta exposure
        if abs(self.risk_metrics.total_theta) > self.config.risk.max_theta_exposure:
            self.alerts.append(RiskAlert(
                alert_type="theta_exposure",
                level=RiskLevel.MEDIUM,
                message=f"Theta exposure {self.risk_metrics.total_theta:.3f} exceeds limit {self.config.risk.max_theta_exposure:.3f}",
                current_value=self.risk_metrics.total_theta,
                limit_value=self.config.risk.max_theta_exposure,
                timestamp=pd.Timestamp.now(),
                action_required=True
            ))
        
        # Check vega exposure
        if abs(self.risk_metrics.total_vega) > self.config.risk.max_vega_exposure:
            self.alerts.append(RiskAlert(
                alert_type="vega_exposure",
                level=RiskLevel.HIGH,
                message=f"Vega exposure {self.risk_metrics.total_vega:.3f} exceeds limit {self.config.risk.max_vega_exposure:.3f}",
                current_value=self.risk_metrics.total_vega,
                limit_value=self.config.risk.max_vega_exposure,
                timestamp=pd.Timestamp.now(),
                action_required=True
            ))
        
        # Check drawdown
        if self.current_drawdown > self.config.risk.max_drawdown:
            self.alerts.append(RiskAlert(
                alert_type="drawdown",
                level=RiskLevel.CRITICAL,
                message=f"Current drawdown {self.current_drawdown:.2%} exceeds limit {self.config.risk.max_drawdown:.2%}",
                current_value=self.current_drawdown,
                limit_value=self.config.risk.max_drawdown,
                timestamp=pd.Timestamp.now(),
                action_required=True
            ))
        
        # Check daily loss
        daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        if daily_pnl < -self.account_value * self.config.risk.max_daily_loss:
            self.alerts.append(RiskAlert(
                alert_type="daily_loss",
                level=RiskLevel.CRITICAL,
                message=f"Daily loss {daily_pnl:.2f} exceeds limit {self.account_value * self.config.risk.max_daily_loss:.2f}",
                current_value=daily_pnl,
                limit_value=-self.account_value * self.config.risk.max_daily_loss,
                timestamp=pd.Timestamp.now(),
                action_required=True
            ))
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        if not hasattr(self, 'risk_metrics'):
            return {}
        
        return {
            'portfolio_metrics': {
                'total_delta': self.risk_metrics.total_delta,
                'total_gamma': self.risk_metrics.total_gamma,
                'total_theta': self.risk_metrics.total_theta,
                'total_vega': self.risk_metrics.total_vega,
                'total_rho': self.risk_metrics.total_rho,
                'total_exposure': self.risk_metrics.total_exposure,
                'max_loss': self.risk_metrics.max_loss,
                'var_95': self.risk_metrics.var_95,
                'var_99': self.risk_metrics.var_99,
                'expected_shortfall': self.risk_metrics.expected_shortfall,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'current_drawdown': self.risk_metrics.current_drawdown
            },
            'position_count': len(self.positions),
            'active_alerts': len(self.alerts),
            'alerts': [
                {
                    'type': alert.alert_type,
                    'level': alert.level.value,
                    'message': alert.message,
                    'action_required': alert.action_required
                }
                for alert in self.alerts
            ],
            'account_value': self.account_value,
            'total_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for pos in self.positions.values():
            data.append({
                'structure_id': pos.structure_id,
                'underlying': pos.underlying,
                'structure_type': pos.structure_type,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'net_delta': pos.net_delta,
                'net_gamma': pos.net_gamma,
                'net_theta': pos.net_theta,
                'net_vega': pos.net_vega,
                'max_loss': pos.max_loss,
                'max_profit': pos.max_profit,
                'expected_value': pos.expected_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'entry_time': pos.entry_time,
                'expiry_time': pos.expiry_time
            })
        
        return pd.DataFrame(data)
    
    def update_account_value(self, new_value: float):
        """Update account value."""
        self.account_value = new_value
        self._update_risk_metrics()
        self._check_risk_alerts()
    
    def close_all_positions(self):
        """Close all positions (emergency stop)."""
        self.positions.clear()
        self._update_risk_metrics()
        self._check_risk_alerts()
        print("All positions closed")
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            'max_delta_exposure': self.config.risk.max_delta_exposure,
            'max_gamma_exposure': self.config.risk.max_gamma_exposure,
            'max_theta_exposure': self.config.risk.max_theta_exposure,
            'max_vega_exposure': self.config.risk.max_vega_exposure,
            'max_loss_per_trade': self.config.risk.max_loss_per_trade,
            'max_daily_loss': self.config.risk.max_daily_loss,
            'max_drawdown': self.config.risk.max_drawdown,
            'max_position_size_pct': self.config.risk.max_position_size_pct,
            'kelly_fraction': self.config.risk.kelly_fraction,
            'min_expected_value': self.config.risk.min_expected_value
        }
