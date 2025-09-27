"""
Strategy Configuration Management

Centralized configuration for the trading strategy including:
- Market parameters (underlyings, expiries, strikes)
- Model parameters (GBT, feature engineering)
- Risk parameters (position sizing, limits)
- Execution parameters (timing, costs)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import time, timedelta
import yaml
import os


@dataclass
class MarketConfig:
    """Market-specific configuration."""
    # Trading horizon and timing
    trading_horizon_days: int = 5  # h: trading horizon in days
    decision_time: str = "15:30:00"  # Daily decision time (15:30 London)
    
    # Target underlyings
    underlyings: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])
    
    # Option selection criteria
    min_days_to_expiry: int = 3  # Minimum days to expiry
    max_days_to_expiry: int = 30  # Maximum days to expiry
    preferred_expiry_days: List[int] = field(default_factory=lambda: [7, 14, 21, 28])  # Weekly expiries
    
    # Strike selection
    strike_range_pct: float = 0.15  # ±15% around ATM
    strike_spacing_pct: float = 0.01  # 1% strike spacing
    min_open_interest: int = 1000  # Minimum OI for liquidity
    max_bid_ask_spread_pct: float = 0.05  # 5% max spread


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Volatility features
    realized_vol_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])  # Days
    bipower_variation_window: int = 20
    jump_detection_threshold: float = 3.0  # Standard deviations
    
    # IV surface features
    iv_surface_points: int = 50  # Number of points for IV surface
    iv_skew_strikes: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])  # Relative strikes
    iv_term_structure_tenors: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90])  # Days
    
    # Cross-asset features
    vix_features: bool = True
    credit_features: bool = True
    rates_features: bool = True
    
    # Volume and liquidity features
    volume_zscore_window: int = 20
    volume_percentile_window: int = 60
    
    # Regime features
    rv_percentile_windows: List[int] = field(default_factory=lambda: [20, 60, 252])  # Days
    regime_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low_vol': 0.25,
        'high_vol': 0.75,
        'extreme_vol': 0.90
    })


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    # GBT parameters
    model_type: str = 'lightgbm'  # 'lightgbm' or 'catboost'
    objective: str = 'classification'  # 'classification' or 'regression'
    
    # LightGBM parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # Cross-validation
    cv_folds: int = 5
    cv_embargo_days: int = 5  # Minimum embargo to avoid overlap leakage
    walk_forward_window: int = 252  # Days for walk-forward validation
    
    # Calibration
    calibration_method: str = 'platt'  # 'platt' or 'isotonic'
    calibration_cv_folds: int = 3


@dataclass
class StructureConfig:
    """Option structure configuration."""
    # Spread parameters
    bull_call_spread_width_pct: float = 0.03  # 3% width for bull call spreads
    bear_put_spread_width_pct: float = 0.03  # 3% width for bear put spreads
    
    # Strike selection
    atm_strike_tolerance_pct: float = 0.01  # 1% tolerance for ATM
    band_coefficient_range: tuple = (0.8, 1.2)  # c ∈ [0.8, 1.2] for forecast band
    
    # Structure selection thresholds
    min_forecast_sharpe: float = 0.3  # k: minimum |S_t| to trade
    max_forecast_sharpe: float = 2.0  # Maximum |S_t| for risk management


@dataclass
class PricingConfig:
    """Pricing and Greeks configuration."""
    # FD solver parameters
    fd_solver: str = 'crank_nicolson'  # 'crank_nicolson' or 'rannacher'
    fd_grid_points: int = 200
    fd_time_steps: int = 200
    fd_s_max_multiple: float = 4.0  # S_max = 4 * K
    
    # IV surface fitting
    iv_surface_model: str = 'svi'  # 'svi' or 'spline'
    iv_smoothing_factor: float = 0.1
    
    # Greeks calculation
    greeks_perturbation_pct: float = 0.01  # 1% perturbation for finite differences
    greeks_time_perturbation: float = 0.01  # 1 day perturbation for theta


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position sizing
    max_position_size_pct: float = 0.01  # 1% of account per trade
    kelly_fraction: float = 0.25  # Kelly fraction for sizing
    max_kelly_fraction: float = 0.5  # Cap on Kelly fraction
    
    # Risk limits
    max_delta_exposure: float = 0.1  # 10% of account in delta
    max_gamma_exposure: float = 0.05  # 5% of account in gamma
    max_theta_exposure: float = 0.02  # 2% of account in theta
    max_vega_exposure: float = 0.1  # 10% of account in vega
    
    # Loss limits
    max_loss_per_trade: float = 0.005  # 0.5% max loss per trade
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_drawdown: float = 0.05  # 5% max drawdown
    
    # Expected value thresholds
    min_expected_value: float = 0.001  # τ: minimum EV threshold (0.1%)


@dataclass
class ExecutionConfig:
    """Execution and costs configuration."""
    # Trading costs
    commission_per_contract: float = 0.65  # $0.65 per contract
    exchange_fees_per_contract: float = 0.10  # $0.10 per contract
    slippage_bps: float = 2.0  # 2 bps slippage
    
    # Order management
    max_order_size: int = 100  # Maximum contracts per order
    order_timeout_seconds: int = 30  # Order timeout
    fill_ratio_threshold: float = 0.8  # Minimum fill ratio
    
    # Risk checks
    pre_trade_checks: bool = True
    real_time_monitoring: bool = True
    auto_hedge_delta: bool = False  # Automatic delta hedging


@dataclass
class StrategyConfig:
    """Main strategy configuration."""
    market: MarketConfig = field(default_factory=MarketConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Strategy metadata
    strategy_name: str = 'OptionsSpreadStrategy'
    version: str = '1.0.0'
    description: str = 'ML-driven options spread trading strategy'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate market config
        assert self.market.trading_horizon_days > 0, "Trading horizon must be positive"
        assert self.market.min_days_to_expiry < self.market.max_days_to_expiry, "Invalid expiry range"
        
        # Validate model config
        assert self.model.objective in ['classification', 'regression'], "Invalid objective"
        assert self.model.model_type in ['lightgbm', 'catboost'], "Invalid model type"
        
        # Validate risk config
        assert 0 < self.risk.kelly_fraction <= 1, "Kelly fraction must be in (0, 1]"
        assert self.risk.max_position_size_pct > 0, "Max position size must be positive"
        
        # Validate structure config
        assert 0 < self.structure.band_coefficient_range[0] < self.structure.band_coefficient_range[1], "Invalid band coefficient range"
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'StrategyConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to config objects
        market_config = MarketConfig(**config_dict.get('market', {}))
        features_config = FeatureConfig(**config_dict.get('features', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        structure_config = StructureConfig(**config_dict.get('structure', {}))
        pricing_config = PricingConfig(**config_dict.get('pricing', {}))
        risk_config = RiskConfig(**config_dict.get('risk', {}))
        execution_config = ExecutionConfig(**config_dict.get('execution', {}))
        
        return cls(
            market=market_config,
            features=features_config,
            model=model_config,
            structure=structure_config,
            pricing=pricing_config,
            risk=risk_config,
            execution=execution_config,
            strategy_name=config_dict.get('strategy_name', 'OptionsSpreadStrategy'),
            version=config_dict.get('version', '1.0.0'),
            description=config_dict.get('description', 'ML-driven options spread trading strategy')
        )
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'strategy_name': self.strategy_name,
            'version': self.version,
            'description': self.description,
            'market': self.market.__dict__,
            'features': self.features.__dict__,
            'model': self.model.__dict__,
            'structure': self.structure.__dict__,
            'pricing': self.pricing.__dict__,
            'risk': self.risk.__dict__,
            'execution': self.execution.__dict__
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_trading_schedule(self, start_date, end_date):
        """Get trading schedule for the given date range."""
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Generate trading days
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Create decision times
        schedule = []
        for date in trading_days:
            # Parse decision time string
            decision_time_str = self.market.decision_time
            if isinstance(decision_time_str, str):
                hour, minute, second = map(int, decision_time_str.split(':'))
                decision_time = datetime.combine(date.date(), time(hour, minute, second))
            else:
                decision_time = datetime.combine(date.date(), decision_time_str)
            
            schedule.append({
                'date': date.date(),
                'decision_time': decision_time,
                'horizon_end': date + timedelta(days=self.market.trading_horizon_days)
            })
        
        return pd.DataFrame(schedule)
    
    def __str__(self):
        """String representation of configuration."""
        return f"""
Strategy Configuration: {self.strategy_name} v{self.version}
===============================================
Market: {len(self.market.underlyings)} underlyings, {self.market.trading_horizon_days}d horizon
Model: {self.model.model_type} ({self.model.objective})
Structure: Bull/Bear spreads with {self.structure.bull_call_spread_width_pct:.1%} width
Pricing: {self.pricing.fd_solver} FD solver
Risk: {self.risk.max_position_size_pct:.1%} max position, {self.risk.kelly_fraction:.1%} Kelly
Execution: ${self.execution.commission_per_contract:.2f} commission, {self.execution.slippage_bps}bps slippage
"""
