# Strategy Framework API Reference

## Table of Contents

1. [Configuration Classes](#configuration-classes)
2. [Feature Engineering](#feature-engineering)
3. [Machine Learning](#machine-learning)
4. [Option Structures](#option-structures)
5. [Pricing Integration](#pricing-integration)
6. [Risk Management](#risk-management)
7. [Backtesting](#backtesting)
8. [Utility Classes](#utility-classes)

## Configuration Classes

### StrategyConfig

Main configuration container for the entire strategy framework.

```python
class StrategyConfig:
    def __init__(self, 
                 market: MarketConfig = None,
                 features: FeatureConfig = None,
                 model: ModelConfig = None,
                 structure: StructureConfig = None,
                 pricing: PricingConfig = None,
                 risk: RiskConfig = None,
                 execution: ExecutionConfig = None,
                 strategy_name: str = "OptionsSpreadStrategy",
                 version: str = "1.0.0"):
        """
        Initialize strategy configuration.
        
        Args:
            market: Market-specific configuration
            features: Feature engineering configuration
            model: ML model configuration
            structure: Option structure configuration
            pricing: FD pricing configuration
            risk: Risk management configuration
            execution: Execution configuration
            strategy_name: Name of the strategy
            version: Strategy version
        """
```

#### Methods

##### `from_yaml(yaml_path: str) -> StrategyConfig`
Load configuration from YAML file.

```python
config = StrategyConfig.from_yaml('config/strategy_config.yaml')
```

**Parameters:**
- `yaml_path`: Path to YAML configuration file

**Returns:**
- `StrategyConfig`: Loaded configuration object

**Raises:**
- `FileNotFoundError`: If YAML file doesn't exist
- `yaml.YAMLError`: If YAML file is invalid

##### `to_yaml(yaml_path: str) -> None`
Save configuration to YAML file.

```python
config.to_yaml('config/my_config.yaml')
```

**Parameters:**
- `yaml_path`: Path to save YAML file

##### `get_trading_schedule(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame`
Generate trading schedule with decision times.

```python
schedule = config.get_trading_schedule(
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31')
)
```

**Parameters:**
- `start_date`: Start date for schedule
- `end_date`: End date for schedule

**Returns:**
- `pd.DataFrame`: Trading schedule with columns:
  - `date`: Trading date
  - `decision_time`: Decision timestamp
  - `horizon_end`: End of trading horizon

### MarketConfig

Market-specific configuration parameters.

```python
@dataclass
class MarketConfig:
    trading_horizon_days: int = 5
    decision_time: str = "15:30:00"
    underlyings: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])
    min_days_to_expiry: int = 3
    max_days_to_expiry: int = 30
    strike_range_pct: float = 0.15
    min_open_interest: int = 1000
```

**Attributes:**
- `trading_horizon_days`: Trading horizon in days
- `decision_time`: Daily decision time (HH:MM:SS format)
- `underlyings`: List of target underlying symbols
- `min_days_to_expiry`: Minimum days to expiry for options
- `max_days_to_expiry`: Maximum days to expiry for options
- `strike_range_pct`: Strike selection range (0.15 = 15%)
- `min_open_interest`: Minimum open interest for liquidity

### ModelConfig

Machine learning model configuration.

```python
@dataclass
class ModelConfig:
    model_type: str = "lightgbm"
    objective: str = "classification"
    lightgbm_params: dict = field(default_factory=dict)
    catboost_params: dict = field(default_factory=dict)
    cv_folds: int = 5
    cv_embargo_days: int = 5
    calibration_method: str = "platt"
```

**Attributes:**
- `model_type`: ML model type ("lightgbm", "catboost", "sklearn")
- `objective`: Prediction objective ("classification", "regression")
- `lightgbm_params`: LightGBM-specific parameters
- `catboost_params`: CatBoost-specific parameters
- `cv_folds`: Number of cross-validation folds
- `cv_embargo_days`: Embargo days to avoid look-ahead bias
- `calibration_method`: Probability calibration method

## Feature Engineering

### FeatureBuilder

Main feature engineering class that builds predictive features from market data.

```python
class FeatureBuilder:
    def __init__(self, config: StrategyConfig):
        """
        Initialize feature builder.
        
        Args:
            config: Strategy configuration
        """
```

#### Methods

##### `build_features(price_data: pd.DataFrame, 
                   options_data: pd.DataFrame = None,
                   vix_data: pd.DataFrame = None,
                   rates_data: pd.DataFrame = None,
                   credit_data: pd.DataFrame = None) -> FeatureSet`
Build comprehensive feature set from market data.

```python
feature_set = feature_builder.build_features(
    price_data=price_data,
    options_data=options_data,
    vix_data=vix_data,
    rates_data=rates_data,
    credit_data=credit_data
)
```

**Parameters:**
- `price_data`: OHLCV price data (required)
- `options_data`: Options chain data (optional)
- `vix_data`: VIX data (optional)
- `rates_data`: Interest rates data (optional)
- `credit_data`: Credit spreads data (optional)

**Returns:**
- `FeatureSet`: Container with features and targets

**Data Requirements:**

**Price Data:**
```python
price_data = pd.DataFrame({
    'open': [100.0, 101.0, ...],
    'high': [101.5, 102.5, ...],
    'low': [99.5, 100.5, ...],
    'close': [101.0, 102.0, ...],
    'volume': [1000000, 1200000, ...]
}, index=pd.DatetimeIndex(['2023-01-01', ...]))
```

**Options Data:**
```python
options_data = pd.DataFrame({
    'iv_atm': [0.15, 0.16, ...],
    'iv_skew_90%': [0.20, 0.21, ...],
    'iv_skew_95%': [0.18, 0.19, ...],
    'iv_skew_100%': [0.15, 0.16, ...],
    'iv_skew_105%': [0.17, 0.18, ...],
    'iv_skew_110%': [0.19, 0.20, ...],
    'iv_7d': [0.15, 0.16, ...],
    'iv_14d': [0.16, 0.17, ...],
    'iv_30d': [0.17, 0.18, ...],
    'iv_60d': [0.18, 0.19, ...],
    'iv_90d': [0.19, 0.20, ...],
    'avg_bid_ask_spread': [0.01, 0.012, ...],
    'total_oi': [10000, 11000, ...]
}, index=pd.DatetimeIndex(['2023-01-01', ...]))
```

### FeatureSet

Container for engineered features and targets.

```python
@dataclass
class FeatureSet:
    features: pd.DataFrame
    target_returns: pd.Series
    target_direction: pd.Series
    feature_names: List[str]
    target_names: List[str]
    metadata: dict
```

**Attributes:**
- `features`: Engineered features DataFrame
- `target_returns`: Target returns series
- `target_direction`: Target direction series (1 for positive, 0 for negative)
- `feature_names`: List of feature names
- `target_names`: List of target names
- `metadata`: Additional metadata

## Machine Learning

### GBTForecaster

Machine learning forecasting class with support for multiple GBT models.

```python
class GBTForecaster:
    def __init__(self, config: StrategyConfig):
        """
        Initialize forecaster.
        
        Args:
            config: Strategy configuration
        """
```

#### Methods

##### `train(X: pd.DataFrame, 
          y: pd.Series,
          sample_weight: Optional[pd.Series] = None) -> ModelResults`
Train the forecasting model.

```python
model_results = forecaster.train(
    X=feature_set.features,
    y=feature_set.target_direction,
    sample_weight=None
)
```

**Parameters:**
- `X`: Feature matrix
- `y`: Target series
- `sample_weight`: Optional sample weights

**Returns:**
- `ModelResults`: Training results with metrics

##### `predict(X: pd.DataFrame) -> ModelResults`
Make predictions on new data.

```python
predictions = forecaster.predict(X_test)
```

**Parameters:**
- `X`: Feature matrix for prediction

**Returns:**
- `ModelResults`: Prediction results

##### `cross_validate(X: pd.DataFrame, 
                   y: pd.Series,
                   sample_weight: Optional[pd.Series] = None) -> dict`
Perform cross-validation.

```python
cv_results = forecaster.cross_validate(
    X=feature_set.features,
    y=feature_set.target_direction
)
```

**Parameters:**
- `X`: Feature matrix
- `y`: Target series
- `sample_weight`: Optional sample weights

**Returns:**
- `dict`: Cross-validation results with metrics

##### `get_forecast(X: pd.DataFrame) -> dict`
Get forecast with confidence measures.

```python
forecast = forecaster.get_forecast(X_test)
```

**Parameters:**
- `X`: Feature matrix for forecasting

**Returns:**
- `dict`: Forecast dictionary with keys:
  - `expected_returns`: Expected return
  - `probabilities`: Probability of positive return
  - `confidence`: Forecast confidence

##### `get_feature_importance() -> pd.DataFrame`
Get feature importance rankings.

```python
importance_df = forecaster.get_feature_importance()
```

**Returns:**
- `pd.DataFrame`: Feature importance with columns:
  - `feature`: Feature name
  - `importance`: Importance score

### ModelResults

Container for model predictions and metrics.

```python
@dataclass
class ModelResults:
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    metrics: Optional[dict] = None
    feature_importance: Optional[pd.DataFrame] = None
    model: Optional[object] = None
```

**Attributes:**
- `predictions`: Model predictions
- `probabilities`: Predicted probabilities (for classification)
- `metrics`: Performance metrics
- `feature_importance`: Feature importance rankings
- `model`: Trained model object

## Option Structures

### OptionStructureSelector

Intelligent option structure selection based on ML forecasts.

```python
class OptionStructureSelector:
    def __init__(self, config: StrategyConfig):
        """
        Initialize structure selector.
        
        Args:
            config: Strategy configuration
        """
```

#### Methods

##### `select_structure(forecast: dict,
                     market_data: dict,
                     current_price: float,
                     current_time: pd.Timestamp) -> StructureSelection`
Select optimal option structure based on forecast.

```python
selection = selector.select_structure(
    forecast=forecast,
    market_data=market_data,
    current_price=100.0,
    current_time=pd.Timestamp.now()
)
```

**Parameters:**
- `forecast`: ML forecast dictionary
- `market_data`: Current market data
- `current_price`: Current underlying price
- `current_time`: Current timestamp

**Returns:**
- `StructureSelection`: Selection result

### OptionStructure

Base class for option structures.

```python
@dataclass
class OptionStructure:
    structure_type: str
    long_leg: Strike
    short_leg: Strike
    expiry: pd.Timestamp
    expected_value: float
    max_profit: float
    max_loss: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
```

**Attributes:**
- `structure_type`: Type of structure ("bull_call_spread", "bear_put_spread")
- `long_leg`: Long option leg
- `short_leg`: Short option leg
- `expiry`: Expiration date
- `expected_value`: Expected value under P-measure
- `max_profit`: Maximum profit
- `max_loss`: Maximum loss
- `net_delta`: Net delta exposure
- `net_gamma`: Net gamma exposure
- `net_theta`: Net theta exposure
- `net_vega`: Net vega exposure
- `net_rho`: Net rho exposure

### Strike

Individual option strike information.

```python
@dataclass
class Strike:
    strike_price: float
    option_type: str
    expiry: pd.Timestamp
    implied_vol: float
    bid_price: float
    ask_price: float
    open_interest: int
    volume: int
```

**Attributes:**
- `strike_price`: Strike price
- `option_type`: Option type ("call", "put")
- `expiry`: Expiration date
- `implied_vol`: Implied volatility
- `bid_price`: Bid price
- `ask_price`: Ask price
- `open_interest`: Open interest
- `volume`: Volume

## Pricing Integration

### FDPricer

Finite difference pricing integration for accurate option pricing.

```python
class FDPricer:
    def __init__(self, config: StrategyConfig):
        """
        Initialize FD pricer.
        
        Args:
            config: Strategy configuration
        """
```

#### Methods

##### `price_option(strike: float,
                 option_type: str,
                 expiry_days: int,
                 current_price: float,
                 risk_free_rate: float,
                 dividend_yield: float,
                 implied_vol: float) -> PricingResult`
Price individual option using FD solver.

```python
pricing = pricer.price_option(
    strike=100.0,
    option_type='call',
    expiry_days=30,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    implied_vol=0.2
)
```

**Parameters:**
- `strike`: Strike price
- `option_type`: Option type ("call", "put")
- `expiry_days`: Days to expiration
- `current_price`: Current underlying price
- `risk_free_rate`: Risk-free interest rate
- `dividend_yield`: Dividend yield
- `implied_vol`: Implied volatility

**Returns:**
- `PricingResult`: Pricing result with Greeks

##### `price_structure(structure: OptionStructure,
                    current_price: float,
                    risk_free_rate: float,
                    dividend_yield: float,
                    implied_vol: float) -> StructurePricing`
Price option structure using FD solver.

```python
structure_pricing = pricer.price_structure(
    structure=structure,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    implied_vol=0.2
)
```

**Parameters:**
- `structure`: Option structure to price
- `current_price`: Current underlying price
- `risk_free_rate`: Risk-free interest rate
- `dividend_yield`: Dividend yield
- `implied_vol`: Implied volatility

**Returns:**
- `StructurePricing`: Structure pricing result

##### `calculate_expected_value(structure: OptionStructure,
                             forecast: dict,
                             current_price: float,
                             risk_free_rate: float,
                             dividend_yield: float) -> float`
Calculate expected value under P-measure.

```python
expected_value = pricer.calculate_expected_value(
    structure=structure,
    forecast=forecast,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0
)
```

**Parameters:**
- `structure`: Option structure
- `forecast`: ML forecast dictionary
- `current_price`: Current underlying price
- `risk_free_rate`: Risk-free interest rate
- `dividend_yield`: Dividend yield

**Returns:**
- `float`: Expected value

### PricingResult

Individual option pricing result.

```python
@dataclass
class PricingResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float
    solver_info: dict
```

**Attributes:**
- `price`: Option price
- `delta`: Delta sensitivity
- `gamma`: Gamma sensitivity
- `theta`: Theta sensitivity
- `vega`: Vega sensitivity
- `rho`: Rho sensitivity
- `iv`: Implied volatility
- `solver_info`: Solver information

## Risk Management

### RiskManager

Comprehensive risk management system.

```python
class RiskManager:
    def __init__(self, config: StrategyConfig):
        """
        Initialize risk manager.
        
        Args:
            config: Strategy configuration
        """
```

#### Methods

##### `check_trade_eligibility(structure: OptionStructure,
                            expected_value: float,
                            current_price: float) -> Tuple[bool, str]`
Check if trade meets risk criteria.

```python
is_eligible, reason = risk_manager.check_trade_eligibility(
    structure=structure,
    expected_value=0.15,
    current_price=100.0
)
```

**Parameters:**
- `structure`: Option structure
- `expected_value`: Expected value
- `current_price`: Current underlying price

**Returns:**
- `Tuple[bool, str]`: (is_eligible, reason)

##### `calculate_position_size(expected_value: float,
                           max_loss: float,
                           current_price: float) -> int`
Calculate optimal position size using Kelly criterion.

```python
position_size = risk_manager.calculate_position_size(
    expected_value=0.15,
    max_loss=250.0,
    current_price=100.0
)
```

**Parameters:**
- `expected_value`: Expected value
- `max_loss`: Maximum loss per contract
- `current_price`: Current underlying price

**Returns:**
- `int`: Position size in contracts

##### `add_position(position: Position) -> bool`
Add position to portfolio.

```python
success = risk_manager.add_position(position)
```

**Parameters:**
- `position`: Position to add

**Returns:**
- `bool`: Success status

##### `get_portfolio_metrics() -> RiskMetrics`
Get current portfolio risk metrics.

```python
metrics = risk_manager.get_portfolio_metrics()
```

**Returns:**
- `RiskMetrics`: Portfolio risk metrics

### Position

Individual position tracking.

```python
@dataclass
class Position:
    position_id: str
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
```

**Attributes:**
- `position_id`: Unique position identifier
- `underlying`: Underlying symbol
- `structure_type`: Structure type
- `quantity`: Position quantity
- `entry_price`: Entry price
- `current_price`: Current price
- `net_delta`: Net delta exposure
- `net_gamma`: Net gamma exposure
- `net_theta`: Net theta exposure
- `net_vega`: Net vega exposure
- `net_rho`: Net rho exposure
- `max_loss`: Maximum loss
- `max_profit`: Maximum profit
- `entry_time`: Entry timestamp
- `expiry_time`: Expiration timestamp
- `expected_value`: Expected value

## Backtesting

### Backtester

Comprehensive backtesting framework.

```python
class Backtester:
    def __init__(self, config: StrategyConfig):
        """
        Initialize backtester.
        
        Args:
            config: Strategy configuration
        """
```

#### Methods

##### `run_backtest(feature_data: pd.DataFrame,
                 market_data: dict,
                 model: object,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp) -> BacktestResults`
Run comprehensive backtest.

```python
results = backtester.run_backtest(
    feature_data=feature_set.features,
    market_data=market_data,
    model=trained_model,
    start_date=pd.Timestamp('2022-01-01'),
    end_date=pd.Timestamp('2023-12-31')
)
```

**Parameters:**
- `feature_data`: Feature data for backtesting
- `market_data`: Market data dictionary
- `model`: Trained ML model
- `start_date`: Backtest start date
- `end_date`: Backtest end date

**Returns:**
- `BacktestResults`: Comprehensive backtest results

### BacktestResults

Comprehensive backtest results.

```python
@dataclass
class BacktestResults:
    trades: List[Trade]
    performance_metrics: dict
    risk_metrics: dict
    trade_analysis: dict
    equity_curve: pd.Series
    drawdown_series: pd.Series
    monthly_returns: pd.Series
```

**Attributes:**
- `trades`: List of executed trades
- `performance_metrics`: Performance metrics dictionary
- `risk_metrics`: Risk metrics dictionary
- `trade_analysis`: Trade analysis dictionary
- `equity_curve`: Equity curve series
- `drawdown_series`: Drawdown series
- `monthly_returns`: Monthly returns series

### Trade

Individual trade record.

```python
@dataclass
class Trade:
    trade_id: str
    timestamp: pd.Timestamp
    underlying: str
    structure_type: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    slippage: float
    net_pnl: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str
    expected_value: float
```

**Attributes:**
- `trade_id`: Unique trade identifier
- `timestamp`: Trade timestamp
- `underlying`: Underlying symbol
- `structure_type`: Structure type
- `quantity`: Trade quantity
- `entry_price`: Entry price
- `exit_price`: Exit price
- `pnl`: Gross P&L
- `commission`: Commission cost
- `slippage`: Slippage cost
- `net_pnl`: Net P&L after costs
- `entry_time`: Entry timestamp
- `exit_time`: Exit timestamp
- `exit_reason`: Exit reason
- `expected_value`: Expected value

## Utility Classes

### ModelResults

Container for model predictions and metrics.

```python
@dataclass
class ModelResults:
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    metrics: Optional[dict] = None
    feature_importance: Optional[pd.DataFrame] = None
    model: Optional[object] = None
```

### StructureSelection

Option structure selection result.

```python
@dataclass
class StructureSelection:
    structure: Optional[OptionStructure]
    selection_reason: str
    forecast_strength: float
    expected_value: float
    confidence: float
```

### PricingResult

Individual option pricing result.

```python
@dataclass
class PricingResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float
    solver_info: dict
```

### StructurePricing

Structure-level pricing result.

```python
@dataclass
class StructurePricing:
    net_price: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    expected_value: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
```

### RiskMetrics

Portfolio risk metrics.

```python
@dataclass
class RiskMetrics:
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    portfolio_value: float
    max_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
```

## Error Handling

### Common Exceptions

#### `StrategyError`
Base exception for strategy framework.

```python
class StrategyError(Exception):
    """Base exception for strategy framework."""
    pass
```

#### `ConfigurationError`
Configuration-related errors.

```python
class ConfigurationError(StrategyError):
    """Configuration error."""
    pass
```

#### `DataError`
Data-related errors.

```python
class DataError(StrategyError):
    """Data error."""
    pass
```

#### `ModelError`
Model-related errors.

```python
class ModelError(StrategyError):
    """Model error."""
    pass
```

#### `PricingError`
Pricing-related errors.

```python
class PricingError(StrategyError):
    """Pricing error."""
    pass
```

#### `RiskError`
Risk-related errors.

```python
class RiskError(StrategyError):
    """Risk error."""
    pass
```

## Usage Examples

### Complete Strategy Pipeline

```python
from strategy import (
    StrategyConfig, FeatureBuilder, GBTForecaster,
    OptionStructureSelector, FDPricer, RiskManager, Backtester
)

# 1. Load configuration
config = StrategyConfig.from_yaml('config/strategy_config.yaml')

# 2. Load market data
market_data = load_market_data()

# 3. Build features
feature_builder = FeatureBuilder(config)
feature_set = feature_builder.build_features(**market_data)

# 4. Train model
forecaster = GBTForecaster(config)
model_results = forecaster.train(feature_set.features, feature_set.target_direction)

# 5. Run backtest
backtester = Backtester(config)
results = backtester.run_backtest(
    feature_data=feature_set.features,
    market_data=market_data,
    model=forecaster.model,
    start_date=pd.Timestamp('2022-01-01'),
    end_date=pd.Timestamp('2023-12-31')
)

# 6. Analyze results
print(f"Total Trades: {results.performance_metrics['total_trades']}")
print(f"Hit Rate: {results.performance_metrics['hit_rate']:.2%}")
print(f"Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.3f}")
```

### Live Trading Example

```python
# Initialize components
feature_builder = FeatureBuilder(config)
forecaster = GBTForecaster(config)
structure_selector = OptionStructureSelector(config)
pricer = FDPricer(config)
risk_manager = RiskManager(config)

# Load trained model
forecaster.model = load_trained_model()

# Get latest data
market_data = get_latest_market_data()

# Build features
feature_set = feature_builder.build_features(**market_data)

# Generate forecast
forecast = forecaster.get_forecast(feature_set.features.iloc[-1:])

# Select structure
selection = structure_selector.select_structure(
    forecast=forecast,
    market_data=market_data,
    current_price=market_data['prices']['close'].iloc[-1],
    current_time=pd.Timestamp.now()
)

# Check risk eligibility
if selection.structure is not None:
    is_eligible, reason = risk_manager.check_trade_eligibility(
        selection.structure,
        selection.expected_value,
        market_data['prices']['close'].iloc[-1]
    )
    
    if is_eligible:
        # Execute trade
        execute_trade(selection.structure)
```

This comprehensive API reference provides detailed documentation for all classes, methods, and parameters in the strategy framework. Each component is designed to work seamlessly together while maintaining flexibility for customization and extension.
