# Options Trading Strategy Framework

## Overview

The Options Trading Strategy Framework is a comprehensive, production-ready system that combines machine learning forecasting (P-measure) with finite difference option pricing (Q-measure) for systematic options spread trading. The framework implements the complete pipeline from data ingestion to trade execution.

## Architecture

```
strategy/
├── config.py          # Configuration management
├── features.py         # Feature engineering pipeline
├── forecasting.py      # ML forecasting models
├── structures.py       # Option structure selection
├── pricing.py          # FD pricing integration
├── risk.py            # Risk management system
├── backtesting.py     # Backtesting framework
└── __init__.py        # Package initialization
```

## Core Components

### 1. Configuration Management (`config.py`)

The configuration system provides centralized parameter management with YAML support.

#### Key Classes

- **`StrategyConfig`**: Main configuration container
- **`MarketConfig`**: Market-specific parameters
- **`FeatureConfig`**: Feature engineering settings
- **`ModelConfig`**: ML model parameters
- **`StructureConfig`**: Option structure settings
- **`PricingConfig`**: FD solver parameters
- **`RiskConfig`**: Risk management limits
- **`ExecutionConfig`**: Trading execution settings

#### Usage Example

```python
from strategy import StrategyConfig

# Load from YAML
config = StrategyConfig.from_yaml('config/strategy_config.yaml')

# Create with defaults
config = StrategyConfig()

# Customize parameters
config.market.trading_horizon_days = 5
config.model.model_type = 'lightgbm'
config.risk.max_position_size_pct = 0.01

# Save configuration
config.to_yaml('my_config.yaml')
```

#### Configuration Parameters

**Market Configuration:**
- `trading_horizon_days`: Trading horizon in days (default: 5)
- `decision_time`: Daily decision time (default: "15:30:00")
- `underlyings`: Target underlyings (default: ["SPY", "QQQ", "IWM"])
- `min_days_to_expiry`: Minimum days to expiry (default: 3)
- `max_days_to_expiry`: Maximum days to expiry (default: 30)
- `strike_range_pct`: Strike selection range (default: 0.15)
- `min_open_interest`: Minimum OI for liquidity (default: 1000)

**Model Configuration:**
- `model_type`: ML model type ("lightgbm", "catboost", "sklearn")
- `objective`: Prediction objective ("classification", "regression")
- `cv_folds`: Cross-validation folds (default: 5)
- `cv_embargo_days`: Embargo days to avoid look-ahead bias (default: 5)
- `calibration_method`: Probability calibration ("platt", "isotonic")

**Risk Configuration:**
- `max_position_size_pct`: Maximum position size (default: 0.01)
- `kelly_fraction`: Kelly criterion fraction (default: 0.25)
- `max_delta_exposure`: Maximum delta exposure (default: 0.1)
- `max_gamma_exposure`: Maximum gamma exposure (default: 0.05)
- `max_drawdown`: Maximum drawdown limit (default: 0.05)

### 2. Feature Engineering (`features.py`)

Comprehensive feature engineering pipeline that builds predictive features from market data.

#### Key Classes

- **`FeatureBuilder`**: Main feature engineering class
- **`FeatureSet`**: Container for engineered features and targets

#### Feature Categories

**Price Features:**
- Returns (1d, 5d, 20d)
- Overnight gaps
- Last hour returns
- Price levels and momentum
- High-low ranges

**Volatility Features:**
- Realized volatility (multiple windows)
- Bipower variation (jump-robust)
- Jump detection and intensity
- Volatility of volatility
- Volatility regimes

**IV Surface Features:**
- ATM IV level and percentiles
- IV skew at different strikes
- IV term structure
- Volatility Risk Premium (VRP)
- IV surface curvature

**Cross-Asset Features:**
- VIX level and term structure
- Interest rates and yield curve
- Credit spreads
- Cross-asset correlations

**Volume and Liquidity Features:**
- Volume z-scores and percentiles
- Options liquidity metrics
- Bid-ask spreads
- Open interest analysis

**Regime Features:**
- Volatility regime indicators
- Trend regime detection
- Momentum regime classification
- Market state dummies

#### Usage Example

```python
from strategy import FeatureBuilder

# Initialize feature builder
feature_builder = FeatureBuilder(config)

# Build features from market data
feature_set = feature_builder.build_features(
    price_data=price_data,
    options_data=options_data,
    vix_data=vix_data,
    rates_data=rates_data,
    credit_data=credit_data
)

# Access features and targets
features = feature_set.features
targets = feature_set.target_returns
feature_names = feature_set.feature_names
```

#### Feature Engineering Process

1. **Data Alignment**: Align all market data to common timestamps
2. **Base Features**: Calculate price-based features (returns, gaps, momentum)
3. **Volatility Features**: Compute realized volatility and jump metrics
4. **IV Features**: Extract implied volatility surface features
5. **Cross-Asset Features**: Add VIX, rates, and credit features
6. **Liquidity Features**: Calculate volume and liquidity metrics
7. **Regime Features**: Identify market regimes and states
8. **Target Construction**: Build forward-looking targets for training

### 3. Machine Learning Forecasting (`forecasting.py`)

Advanced ML forecasting system with multiple model support and robust validation.

#### Key Classes

- **`GBTForecaster`**: Main forecasting class
- **`ModelResults`**: Container for model predictions and metrics

#### Supported Models

**LightGBM:**
- Fast gradient boosting
- Built-in categorical support
- Early stopping and regularization
- Feature importance analysis

**CatBoost:**
- Categorical boosting
- Robust to overfitting
- Built-in categorical encoding
- Advanced regularization

**Scikit-learn Fallback:**
- Random Forest (classification/regression)
- Logistic Regression
- Linear Regression
- Automatic fallback when advanced models unavailable

#### Cross-Validation

**Purged Time Series Split:**
- Avoids look-ahead bias
- Embargo periods between train/validation
- Horizon-aware splits
- Walk-forward validation support

#### Usage Example

```python
from strategy import GBTForecaster

# Initialize forecaster
forecaster = GBTForecaster(config)

# Train model
model_results = forecaster.train(X_train, y_train)

# Cross-validation
cv_results = forecaster.cross_validate(X_train, y_train)

# Make predictions
predictions = forecaster.predict(X_test)

# Get forecasts with confidence
forecast = forecaster.get_forecast(X_test)
```

#### Model Training Process

1. **Data Preparation**: Clean and prepare training data
2. **Model Selection**: Choose model type based on availability
3. **Training**: Fit model with early stopping
4. **Calibration**: Calibrate probabilities (classification)
5. **Validation**: Cross-validate with purged splits
6. **Evaluation**: Calculate comprehensive metrics

#### Performance Metrics

**Classification Metrics:**
- Accuracy and hit rate
- AUC (Area Under Curve)
- Log loss and Brier score
- Information Ratio
- Calibration metrics

**Regression Metrics:**
- MSE, RMSE, MAE
- R-squared
- Information Ratio
- Residual analysis

### 4. Option Structure Selection (`structures.py`)

Intelligent option structure selection based on ML forecasts and market conditions.

#### Key Classes

- **`OptionStructureSelector`**: Main structure selection class
- **`OptionStructure`**: Container for selected structures
- **`Strike`**: Individual option strike information
- **`StructureSelection`**: Selection result with metadata

#### Structure Types

**Bull Call Spreads:**
- Long call at lower strike
- Short call at higher strike
- Limited profit, limited loss
- Bullish directional bet

**Bear Put Spreads:**
- Long put at higher strike
- Short put at lower strike
- Limited profit, limited loss
- Bearish directional bet

#### Selection Process

1. **Forecast Analysis**: Evaluate ML forecast strength
2. **Direction Determination**: Identify bullish/bearish signals
3. **Strike Selection**: Choose strikes based on forecast bands
4. **Liquidity Filtering**: Ensure adequate liquidity
5. **Risk Validation**: Check risk limits and constraints

#### Usage Example

```python
from strategy import OptionStructureSelector

# Initialize selector
selector = OptionStructureSelector(config)

# Select structure
selection = selector.select_structure(
    forecast=forecast,
    market_data=market_data,
    current_price=100.0,
    current_time=pd.Timestamp.now()
)

# Check selection result
if selection.structure is not None:
    structure = selection.structure
    print(f"Selected: {structure.structure_type}")
    print(f"Strikes: {structure.long_leg.strike_price}, {structure.short_leg.strike_price}")
    print(f"Expected Value: {structure.expected_value}")
```

#### Strike Selection Logic

**Forecast Band Calculation:**
```
S_min = S_0 * (1 + μ̂ - c * σ̂)
S_max = S_0 * (1 + μ̂ + c * σ̂)
```

Where:
- `μ̂` = Expected return from ML model
- `σ̂` = Forecasted volatility
- `c` = Band coefficient (0.8 to 1.2)

**Bull Call Spread:**
- Long strike: ATM or slightly ITM
- Short strike: Near upper forecast band
- Width: Typically 3% of underlying price

**Bear Put Spread:**
- Long strike: ATM or slightly ITM
- Short strike: Near lower forecast band
- Width: Typically 3% of underlying price

### 5. FD Pricing Integration (`pricing.py`)

Integration with finite difference solvers for accurate option pricing and Greeks calculation.

#### Key Classes

- **`FDPricer`**: Main pricing class
- **`PricingResult`**: Individual option pricing result
- **`StructurePricing`**: Structure-level pricing result

#### Supported Solvers

**Crank-Nicolson Solver:**
- High accuracy and stability
- Second-order convergence
- Suitable for most applications

**Rannacher Smoothing:**
- Enhanced accuracy near expiry
- Implicit Euler for first few steps
- Crank-Nicolson for remaining steps

#### Pricing Features

- **Option Pricing**: Accurate Q-measure pricing
- **Greeks Calculation**: Delta, gamma, theta, vega, rho
- **IV Surface Fitting**: SVI or spline interpolation
- **Expected Value**: P-measure expected value calculation
- **Risk Metrics**: Comprehensive risk analysis

#### Usage Example

```python
from strategy import FDPricer

# Initialize pricer
pricer = FDPricer(config)

# Price individual option
pricing = pricer.price_option(
    strike=100.0,
    option_type='call',
    expiry_days=30,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    implied_vol=0.2
)

# Price option structure
structure_pricing = pricer.price_structure(
    structure=structure,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    implied_vol=0.2
)

# Calculate expected value
expected_value = pricer.calculate_expected_value(
    structure=structure,
    forecast=forecast,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0
)
```

#### Pricing Process

1. **Solver Selection**: Choose FD solver (CN or Rannacher)
2. **Grid Setup**: Create spatial and temporal grids
3. **Boundary Conditions**: Apply appropriate boundary conditions
4. **Time Stepping**: Solve PDE backwards from expiry
5. **Greeks Calculation**: Compute sensitivities using finite differences
6. **Expected Value**: Calculate P-measure expected value

#### Greeks Calculation

**Delta (Δ):** `∂V/∂S`
**Gamma (Γ):** `∂²V/∂S²`
**Theta (Θ):** `∂V/∂t`
**Vega (ν):** `∂V/∂σ`
**Rho (ρ):** `∂V/∂r`

All Greeks calculated using finite difference methods with appropriate perturbations.

### 6. Risk Management (`risk.py`)

Comprehensive risk management system with position sizing, limits, and monitoring.

#### Key Classes

- **`RiskManager`**: Main risk management class
- **`Position`**: Individual position tracking
- **`RiskMetrics`**: Portfolio risk metrics
- **`RiskAlert`**: Risk limit violation alerts

#### Risk Features

**Position Sizing:**
- Kelly criterion implementation
- Maximum position size limits
- Account value-based sizing
- Risk-adjusted position sizing

**Risk Limits:**
- Delta, gamma, theta, vega exposure limits
- Maximum loss per trade
- Daily loss limits
- Maximum drawdown limits

**Portfolio Monitoring:**
- Real-time risk metrics
- Risk alert system
- Position tracking
- Performance monitoring

#### Usage Example

```python
from strategy import RiskManager, Position

# Initialize risk manager
risk_manager = RiskManager(config)

# Create position
position = Position(
    structure_id="trade_001",
    underlying="SPY",
    structure_type="bull_call_spread",
    quantity=10,
    entry_price=2.50,
    current_price=2.75,
    net_delta=0.5,
    net_gamma=0.02,
    net_theta=-0.1,
    net_vega=0.3,
    net_rho=0.05,
    max_loss=250.0,
    max_profit=750.0,
    entry_time=pd.Timestamp.now(),
    expiry_time=pd.Timestamp.now() + pd.Timedelta(days=30),
    expected_value=0.15
)

# Add position
success = risk_manager.add_position(position)

# Check trade eligibility
is_eligible, reason = risk_manager.check_trade_eligibility(
    structure=structure,
    expected_value=0.15,
    current_price=100.0
)

# Calculate position size
position_size = risk_manager.calculate_position_size(
    expected_value=0.15,
    max_loss=250.0,
    current_price=100.0
)
```

#### Risk Management Process

1. **Pre-Trade Checks**: Validate trade eligibility
2. **Position Sizing**: Calculate optimal position size
3. **Risk Limits**: Check portfolio-level limits
4. **Position Tracking**: Monitor active positions
5. **Risk Monitoring**: Continuous risk assessment
6. **Alert System**: Generate risk alerts
7. **Position Management**: Handle exits and adjustments

#### Kelly Criterion Implementation

```
Position Size = (Expected Value / Max Loss) × Kelly Fraction
```

Where:
- Expected Value = P-measure expected value
- Max Loss = Maximum loss per contract
- Kelly Fraction = Risk-adjusted fraction (default: 0.25)

### 7. Backtesting Framework (`backtesting.py`)

Comprehensive backtesting system with realistic execution simulation.

#### Key Classes

- **`Backtester`**: Main backtesting engine
- **`Trade`**: Individual trade record
- **`BacktestResults`**: Comprehensive backtest results

#### Backtesting Features

**Event Alignment:**
- Proper timestamp alignment
- No look-ahead bias
- Realistic data availability

**Execution Simulation:**
- Commission costs
- Slippage modeling
- Market impact
- Fill ratio simulation

**Performance Analysis:**
- Comprehensive metrics
- Risk-adjusted returns
- Trade analysis
- Regime-based performance

#### Usage Example

```python
from strategy import Backtester

# Initialize backtester
backtester = Backtester(config)

# Run backtest
results = backtester.run_backtest(
    feature_data=feature_data,
    market_data=market_data,
    model=trained_model,
    start_date=pd.Timestamp('2020-01-01'),
    end_date=pd.Timestamp('2023-12-31')
)

# Analyze results
print(f"Total Trades: {results.performance_metrics['total_trades']}")
print(f"Hit Rate: {results.performance_metrics['hit_rate']:.2%}")
print(f"Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results.performance_metrics['max_drawdown']:.2%}")
```

#### Backtesting Process

1. **Data Preparation**: Align historical data
2. **Trading Schedule**: Generate decision times
3. **Feature Building**: Build features for each decision point
4. **Forecasting**: Generate ML forecasts
5. **Structure Selection**: Select option structures
6. **Pricing**: Price structures using FD solvers
7. **Risk Management**: Apply risk controls
8. **Trade Execution**: Simulate realistic execution
9. **Position Management**: Handle exits and monitoring
10. **Performance Analysis**: Calculate comprehensive metrics

#### Performance Metrics

**Trading Metrics:**
- Total trades and hit rate
- Average win/loss
- Profit factor
- Commission and slippage costs

**Risk Metrics:**
- Maximum drawdown
- VaR (95%, 99%)
- Volatility and Sharpe ratio
- Expected shortfall

**Trade Analysis:**
- Performance by structure type
- Exit reason analysis
- Regime-based performance
- Capacity analysis

## Integration Points

### FD Solver Integration

The strategy framework seamlessly integrates with the PDE solvers:

```python
# In pricing.py
from pde import BlackScholesCNSolver, BlackScholesCNRannacherSolver

# Solver selection based on configuration
if config.pricing.fd_solver == 'crank_nicolson':
    solver = BlackScholesCNSolver(...)
elif config.pricing.fd_solver == 'rannacher':
    solver = BlackScholesCNRannacherSolver(...)
```

### Data Integration

The framework is designed to work with various data sources:

```python
# Market data structure
market_data = {
    'prices': price_data,      # OHLCV data
    'options': options_data,   # Options chain data
    'vix': vix_data,          # VIX data
    'rates': rates_data,      # Interest rates
    'credit': credit_data     # Credit spreads
}
```

## Configuration Examples

### Basic Configuration

```yaml
# config/basic_strategy.yaml
strategy_name: "BasicOptionsStrategy"
version: "1.0.0"

market:
  trading_horizon_days: 5
  decision_time: "15:30:00"
  underlyings: ["SPY"]

model:
  model_type: "lightgbm"
  objective: "classification"

risk:
  max_position_size_pct: 0.01
  kelly_fraction: 0.25
  max_drawdown: 0.05
```

### Advanced Configuration

```yaml
# config/advanced_strategy.yaml
strategy_name: "AdvancedOptionsStrategy"
version: "2.0.0"

market:
  trading_horizon_days: 7
  decision_time: "16:00:00"
  underlyings: ["SPY", "QQQ", "IWM"]
  min_days_to_expiry: 5
  max_days_to_expiry: 45

features:
  realized_vol_windows: [5, 10, 20, 60]
  iv_skew_strikes: [0.9, 0.95, 1.0, 1.05, 1.1]
  vix_features: true
  credit_features: true

model:
  model_type: "lightgbm"
  objective: "classification"
  lightgbm_params:
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.9
  cv_folds: 5
  cv_embargo_days: 7

structure:
  bull_call_spread_width_pct: 0.04
  bear_put_spread_width_pct: 0.04
  min_forecast_sharpe: 0.4

pricing:
  fd_solver: "rannacher"
  fd_grid_points: 300
  fd_time_steps: 300

risk:
  max_position_size_pct: 0.015
  kelly_fraction: 0.3
  max_delta_exposure: 0.15
  max_gamma_exposure: 0.08
  max_drawdown: 0.08

execution:
  commission_per_contract: 0.65
  slippage_bps: 1.5
  max_order_size: 200
```

## Error Handling and Robustness

### Graceful Degradation

The framework includes comprehensive error handling:

```python
# ML model fallback
if config.model.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
    print("Warning: LightGBM not available, falling back to scikit-learn")
    config.model.model_type = 'sklearn'

# Feature engineering robustness
try:
    features = feature_builder.build_features(...)
except Exception as e:
    print(f"Feature engineering failed: {e}")
    # Use simplified features or skip

# Pricing fallback
try:
    pricing = pricer.price_option(...)
except Exception as e:
    print(f"Pricing failed: {e}")
    # Use analytical approximation
```

### Data Validation

```python
# Input validation
def validate_market_data(market_data):
    required_keys = ['prices']
    for key in required_keys:
        if key not in market_data:
            raise ValueError(f"Missing required data: {key}")
    
    # Validate data quality
    if market_data['prices'].empty:
        raise ValueError("Price data is empty")
    
    # Check for missing values
    if market_data['prices'].isnull().any().any():
        print("Warning: Price data contains missing values")
```

## Performance Optimization

### Caching

```python
# Solver caching in pricing
class FDPricer:
    def __init__(self, config):
        self.solver_cache = {}
    
    def price_option(self, ...):
        solver_key = f"{option_type}_{strike}_{expiry_days}_{implied_vol}"
        if solver_key not in self.solver_cache:
            solver = self._create_solver(...)
            self.solver_cache[solver_key] = solver
        else:
            solver = self.solver_cache[solver_key]
```

### Parallel Processing

```python
# Parallel feature engineering
from concurrent.futures import ThreadPoolExecutor

def build_features_parallel(self, market_data):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(self._build_price_features, market_data['prices']),
            executor.submit(self._build_volatility_features, market_data['prices']),
            executor.submit(self._build_iv_features, market_data['options'], market_data['prices']),
            executor.submit(self._build_cross_asset_features, ...)
        }
        
        results = [future.result() for future in futures]
        return pd.concat(results, axis=1)
```

## Testing and Validation

### Unit Tests

```python
# Example unit test
def test_feature_builder():
    config = StrategyConfig()
    feature_builder = FeatureBuilder(config)
    
    # Create sample data
    price_data = create_sample_price_data()
    
    # Test feature building
    feature_set = feature_builder.build_features(price_data=price_data)
    
    # Assertions
    assert len(feature_set.features) > 0
    assert len(feature_set.feature_names) > 0
    assert not feature_set.features.isnull().all().all()
```

### Integration Tests

```python
def test_strategy_integration():
    config = StrategyConfig()
    
    # Test complete pipeline
    feature_builder = FeatureBuilder(config)
    forecaster = GBTForecaster(config)
    structure_selector = OptionStructureSelector(config)
    pricer = FDPricer(config)
    risk_manager = RiskManager(config)
    
    # Test data flow
    market_data = create_test_market_data()
    feature_set = feature_builder.build_features(**market_data)
    
    # Test model training
    model_results = forecaster.train(feature_set.features, feature_set.target_direction)
    assert model_results.metrics['accuracy'] > 0.5
```

## Monitoring and Logging

### Comprehensive Logging

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('strategy')

# Usage in components
class GBTForecaster:
    def train(self, X, y):
        logger.info(f"Training model with {len(X)} samples, {len(X.columns)} features")
        # ... training logic
        logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.3f}")
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Usage
@monitor_performance
def build_features(self, market_data):
    # Feature building logic
    pass
```

## Future Extensions

### Additional Models

```python
# XGBoost support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Neural network support
try:
    import torch
    import torch.nn as nn
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError:
    NEURAL_NETWORKS_AVAILABLE = False
```

### Additional Structures

```python
# Iron Condor
class IronCondorStructure(OptionStructure):
    def __init__(self, strikes, expiry):
        self.long_put = strikes['long_put']
        self.short_put = strikes['short_put']
        self.short_call = strikes['short_call']
        self.long_call = strikes['long_call']
        # ... implementation

# Butterfly Spread
class ButterflyStructure(OptionStructure):
    def __init__(self, strikes, expiry):
        self.long_wing = strikes['long_wing']
        self.short_center = strikes['short_center']
        self.long_wing2 = strikes['long_wing2']
        # ... implementation
```

### Real-time Trading

```python
# Real-time data integration
class RealTimeDataFeed:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data_stream = None
    
    def start_stream(self):
        # WebSocket or API connection
        pass
    
    def get_latest_data(self):
        # Return latest market data
        pass

# Real-time strategy execution
class LiveStrategy:
    def __init__(self, config):
        self.config = config
        self.data_feed = RealTimeDataFeed(config.market.underlyings)
        self.strategy_components = self._initialize_components()
    
    def run_live(self):
        while True:
            # Get latest data
            market_data = self.data_feed.get_latest_data()
            
            # Run strategy
            self._process_decision(market_data)
            
            # Wait for next decision time
            time.sleep(self._get_sleep_time())
```

This comprehensive documentation covers all aspects of the strategy framework, from individual components to integration patterns and future extensions. The framework provides a solid foundation for systematic options trading with proper risk management and realistic execution simulation.
