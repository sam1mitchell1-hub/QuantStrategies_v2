# Strategy Framework Examples and Tutorials

## Quick Start Examples

### 1. Basic Strategy Execution

```python
#!/usr/bin/env python3
"""
Basic strategy execution example
"""

from strategy import StrategyConfig, FeatureBuilder, GBTForecaster, Backtester
import pandas as pd

def main():
    # Load configuration
    config = StrategyConfig.from_yaml('config/strategy_config.yaml')
    
    # Load sample data
    market_data = load_sample_data()
    
    # Build features
    feature_builder = FeatureBuilder(config)
    feature_set = feature_builder.build_features(**market_data)
    
    # Train model
    forecaster = GBTForecaster(config)
    model_results = forecaster.train(feature_set.features, feature_set.target_direction)
    
    # Run backtest
    backtester = Backtester(config)
    results = backtester.run_backtest(
        feature_data=feature_set.features,
        market_data=market_data,
        model=forecaster.model,
        start_date=pd.Timestamp('2022-01-01'),
        end_date=pd.Timestamp('2023-12-31')
    )
    
    # Print results
    print(f"Total Trades: {results.performance_metrics['total_trades']}")
    print(f"Hit Rate: {results.performance_metrics['hit_rate']:.2%}")
    print(f"Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.3f}")

if __name__ == "__main__":
    main()
```

### 2. Custom Configuration

```python
from strategy import StrategyConfig, MarketConfig, ModelConfig, RiskConfig

# Create custom configuration
config = StrategyConfig(
    market=MarketConfig(
        trading_horizon_days=7,
        decision_time="16:00:00",
        underlyings=["SPY", "QQQ"],
        min_days_to_expiry=5,
        max_days_to_expiry=45
    ),
    model=ModelConfig(
        model_type="lightgbm",
        objective="classification",
        lightgbm_params={
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9
        }
    ),
    risk=RiskConfig(
        max_position_size_pct=0.01,
        kelly_fraction=0.25,
        max_drawdown=0.05
    )
)

# Save configuration
config.to_yaml('config/my_custom_config.yaml')
```

### 3. Feature Engineering

```python
from strategy import FeatureBuilder
import pandas as pd

# Initialize feature builder
feature_builder = FeatureBuilder(config)

# Build features from market data
feature_set = feature_builder.build_features(
    price_data=price_data,
    options_data=options_data,
    vix_data=vix_data,
    rates_data=rates_data
)

# Access results
features = feature_set.features
targets = feature_set.target_direction
feature_names = feature_set.feature_names

print(f"Built {len(feature_names)} features")
print(f"Feature names: {feature_names[:10]}")  # First 10 features
```

### 4. Model Training and Validation

```python
from strategy import GBTForecaster
from sklearn.model_selection import train_test_split

# Initialize forecaster
forecaster = GBTForecaster(config)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    feature_set.features, 
    feature_set.target_direction, 
    test_size=0.2, 
    random_state=42
)

# Train model
model_results = forecaster.train(X_train, y_train)

# Cross-validation
cv_results = forecaster.cross_validate(X_train, y_train)

# Make predictions
predictions = forecaster.predict(X_test)

# Get feature importance
importance_df = forecaster.get_feature_importance()
print("Top 10 Features:")
print(importance_df.head(10))
```

### 5. Option Structure Selection

```python
from strategy import OptionStructureSelector

# Initialize selector
selector = OptionStructureSelector(config)

# Select structure based on forecast
selection = selector.select_structure(
    forecast={
        'expected_returns': 0.02,
        'probabilities': 0.65,
        'confidence': 0.8
    },
    market_data=market_data,
    current_price=100.0,
    current_time=pd.Timestamp.now()
)

# Check selection
if selection.structure is not None:
    structure = selection.structure
    print(f"Selected: {structure.structure_type}")
    print(f"Long strike: {structure.long_leg.strike_price}")
    print(f"Short strike: {structure.short_leg.strike_price}")
    print(f"Expected value: {structure.expected_value:.4f}")
else:
    print("No structure selected")
```

### 6. FD Pricing Integration

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

print(f"Option price: {pricing.price:.4f}")
print(f"Delta: {pricing.delta:.4f}")
print(f"Gamma: {pricing.gamma:.4f}")
print(f"Theta: {pricing.theta:.4f}")
print(f"Vega: {pricing.vega:.4f}")

# Price option structure
structure_pricing = pricer.price_structure(
    structure=structure,
    current_price=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    implied_vol=0.2
)

print(f"Structure price: {structure_pricing.net_price:.4f}")
print(f"Net delta: {structure_pricing.net_delta:.4f}")
print(f"Expected value: {structure_pricing.expected_value:.4f}")
```

### 7. Risk Management

```python
from strategy import RiskManager, Position

# Initialize risk manager
risk_manager = RiskManager(config)

# Check trade eligibility
is_eligible, reason = risk_manager.check_trade_eligibility(
    structure=structure,
    expected_value=0.15,
    current_price=100.0
)

print(f"Trade eligible: {is_eligible}")
if not is_eligible:
    print(f"Reason: {reason}")

# Calculate position size
position_size = risk_manager.calculate_position_size(
    expected_value=0.15,
    max_loss=250.0,
    current_price=100.0
)

print(f"Position size: {position_size} contracts")

# Add position
position = Position(
    position_id="trade_001",
    underlying="SPY",
    structure_type="bull_call_spread",
    quantity=position_size,
    entry_price=2.50,
    current_price=2.50,
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

success = risk_manager.add_position(position)
print(f"Position added: {success}")
```

### 8. Backtesting

```python
from strategy import Backtester

# Initialize backtester
backtester = Backtester(config)

# Run backtest
results = backtester.run_backtest(
    feature_data=feature_set.features,
    market_data=market_data,
    model=forecaster.model,
    start_date=pd.Timestamp('2022-01-01'),
    end_date=pd.Timestamp('2023-12-31')
)

# Analyze results
print("=== BACKTEST RESULTS ===")
print(f"Total Trades: {results.performance_metrics['total_trades']}")
print(f"Winning Trades: {results.performance_metrics['winning_trades']}")
print(f"Losing Trades: {results.performance_metrics['losing_trades']}")
print(f"Hit Rate: {results.performance_metrics['hit_rate']:.2%}")
print(f"Total P&L: ${results.performance_metrics['total_pnl']:.2f}")
print(f"Total Return: {results.performance_metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results.performance_metrics['max_drawdown']:.2%}")
print(f"Profit Factor: {results.performance_metrics['profit_factor']:.2f}")

# Trade analysis
print("\n=== TRADE ANALYSIS ===")
for structure_type, analysis in results.trade_analysis['structure_analysis'].items():
    print(f"{structure_type}:")
    print(f"  Count: {analysis['count']}")
    print(f"  Hit Rate: {analysis['hit_rate']:.2%}")
    print(f"  Avg P&L: ${analysis['avg_pnl']:.2f}")
```

## Advanced Examples

### 1. Custom Feature Engineering

```python
class CustomFeatureBuilder(FeatureBuilder):
    def _build_custom_features(self, price_data):
        """Build custom features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Custom momentum indicator
        features['custom_momentum'] = price_data['close'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # Custom volatility regime
        returns = price_data['close'].pct_change()
        vol_20d = returns.rolling(20).std()
        features['vol_regime'] = (vol_20d > vol_20d.rolling(60).quantile(0.8)).astype(int)
        
        return features
    
    def build_features(self, **kwargs):
        """Override to include custom features."""
        # Get base features
        feature_set = super().build_features(**kwargs)
        
        # Add custom features
        custom_features = self._build_custom_features(kwargs['price_data'])
        feature_set.features = pd.concat([feature_set.features, custom_features], axis=1)
        
        return feature_set

# Use custom feature builder
custom_feature_builder = CustomFeatureBuilder(config)
feature_set = custom_feature_builder.build_features(**market_data)
```

### 2. Model Ensemble

```python
from strategy import GBTForecaster
import numpy as np

class EnsembleForecaster:
    def __init__(self, config):
        self.config = config
        self.models = {}
        
    def train_ensemble(self, X, y):
        """Train ensemble of models."""
        # Train LightGBM
        config_lightgbm = self.config.copy()
        config_lightgbm.model.model_type = "lightgbm"
        self.models['lightgbm'] = GBTForecaster(config_lightgbm)
        self.models['lightgbm'].train(X, y)
        
        # Train CatBoost (if available)
        try:
            config_catboost = self.config.copy()
            config_catboost.model.model_type = "catboost"
            self.models['catboost'] = GBTForecaster(config_catboost)
            self.models['catboost'].train(X, y)
        except:
            pass
        
        # Train scikit-learn
        config_sklearn = self.config.copy()
        config_sklearn.model.model_type = "sklearn"
        self.models['sklearn'] = GBTForecaster(config_sklearn)
        self.models['sklearn'].train(X, y)
    
    def predict_ensemble(self, X):
        """Make ensemble predictions."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred.probabilities)
            # Simple weighting scheme
            weights.append(1.0 / len(self.models))
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return {
            'probabilities': ensemble_pred,
            'expected_returns': (ensemble_pred - 0.5) * 0.02,
            'confidence': np.abs(ensemble_pred - 0.5) * 2
        }

# Use ensemble
ensemble = EnsembleForecaster(config)
ensemble.train_ensemble(feature_set.features, feature_set.target_direction)
forecast = ensemble.predict_ensemble(X_test)
```

### 3. Real-Time Data Integration

```python
import yfinance as yf
import time

class RealTimeDataManager:
    def __init__(self, symbols):
        self.symbols = symbols
        self.last_update = None
        
    def get_latest_data(self):
        """Get latest market data."""
        current_data = {}
        
        # Get price data
        price_data = yf.download(self.symbols, period="1y", interval="1d")
        current_data['prices'] = price_data
        
        # Get VIX data
        vix_data = yf.download("^VIX", period="1y", interval="1d")
        current_data['vix'] = vix_data
        
        # Get rates data (simplified)
        rates_data = pd.DataFrame({
            '10y_rate': [0.05] * len(price_data),
            '2y_rate': [0.04] * len(price_data)
        }, index=price_data.index)
        current_data['rates'] = rates_data
        
        self.last_update = pd.Timestamp.now()
        return current_data

# Real-time strategy execution
def run_real_time_strategy():
    data_manager = RealTimeDataManager(["SPY", "QQQ"])
    
    # Initialize strategy components
    feature_builder = FeatureBuilder(config)
    forecaster = GBTForecaster(config)
    forecaster.model = load_trained_model()  # Load pre-trained model
    
    while True:
        try:
            # Get latest data
            market_data = data_manager.get_latest_data()
            
            # Check if it's decision time
            current_time = pd.Timestamp.now().time()
            decision_time = pd.to_datetime(config.market.decision_time).time()
            
            if abs((current_time.hour * 60 + current_time.minute) - 
                   (decision_time.hour * 60 + decision_time.minute)) <= 5:
                
                # Build features
                feature_set = feature_builder.build_features(**market_data)
                
                # Generate forecast
                forecast = forecaster.get_forecast(feature_set.features.iloc[-1:])
                
                # Process decision
                process_trading_decision(forecast, market_data)
            
            # Wait 1 minute
            time.sleep(60)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

# Run real-time strategy
# run_real_time_strategy()
```

### 4. Performance Monitoring

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(results):
    """Create comprehensive performance dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Equity curve
    axes[0, 0].plot(results.equity_curve.index, results.equity_curve)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('Cumulative P&L')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    axes[0, 1].fill_between(results.drawdown_series.index, 
                           results.drawdown_series, 0, 
                           alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Monthly returns
    monthly_returns = results.monthly_returns
    axes[0, 2].bar(range(len(monthly_returns)), monthly_returns)
    axes[0, 2].set_title('Monthly Returns')
    axes[0, 2].set_ylabel('Return')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Trade distribution
    trade_pnls = [trade.net_pnl for trade in results.trades]
    axes[1, 0].hist(trade_pnls, bins=30, alpha=0.7)
    axes[1, 0].set_title('Trade P&L Distribution')
    axes[1, 0].set_xlabel('P&L')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hit rate by month
    monthly_hit_rates = results.trade_analysis['monthly_analysis']['hit_rate']
    axes[1, 1].plot(monthly_hit_rates.index, monthly_hit_rates)
    axes[1, 1].set_title('Monthly Hit Rate')
    axes[1, 1].set_ylabel('Hit Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Feature importance
    if hasattr(forecaster, 'get_feature_importance'):
        importance_df = forecaster.get_feature_importance()
        top_features = importance_df.head(10)
        axes[1, 2].barh(range(len(top_features)), top_features['importance'])
        axes[1, 2].set_yticks(range(len(top_features)))
        axes[1, 2].set_yticklabels(top_features['feature'])
        axes[1, 2].set_title('Top 10 Features')
        axes[1, 2].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('output/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create dashboard
create_performance_dashboard(results)
```

## Configuration Examples

### 1. Conservative Strategy

```yaml
# config/conservative_strategy.yaml
strategy_name: "ConservativeOptionsStrategy"
version: "1.0.0"

market:
  trading_horizon_days: 7
  decision_time: "15:30:00"
  underlyings: ["SPY"]
  min_days_to_expiry: 7
  max_days_to_expiry: 21
  strike_range_pct: 0.10
  min_open_interest: 2000

model:
  model_type: "lightgbm"
  objective: "classification"
  lightgbm_params:
    num_leaves: 15
    learning_rate: 0.03
    feature_fraction: 0.8
    bagging_fraction: 0.8
  cv_folds: 5
  cv_embargo_days: 7

structure:
  bull_call_spread_width_pct: 0.02
  bear_put_spread_width_pct: 0.02
  min_forecast_sharpe: 0.6
  band_coefficient_range: [1.0, 1.2]

risk:
  max_position_size_pct: 0.005
  kelly_fraction: 0.1
  max_delta_exposure: 0.05
  max_gamma_exposure: 0.02
  max_drawdown: 0.03
  min_expected_value: 0.002

pricing:
  fd_solver: "crank_nicolson"
  fd_grid_points: 150
  fd_time_steps: 150
  fd_s_max_multiple: 3.0
```

### 2. Aggressive Strategy

```yaml
# config/aggressive_strategy.yaml
strategy_name: "AggressiveOptionsStrategy"
version: "1.0.0"

market:
  trading_horizon_days: 3
  decision_time: "16:00:00"
  underlyings: ["SPY", "QQQ", "IWM"]
  min_days_to_expiry: 3
  max_days_to_expiry: 45
  strike_range_pct: 0.20
  min_open_interest: 500

model:
  model_type: "lightgbm"
  objective: "classification"
  lightgbm_params:
    num_leaves: 63
    learning_rate: 0.08
    feature_fraction: 0.9
    bagging_fraction: 0.9
  cv_folds: 3
  cv_embargo_days: 3

structure:
  bull_call_spread_width_pct: 0.05
  bear_put_spread_width_pct: 0.05
  min_forecast_sharpe: 0.2
  band_coefficient_range: [0.8, 1.2]

risk:
  max_position_size_pct: 0.02
  kelly_fraction: 0.4
  max_delta_exposure: 0.15
  max_gamma_exposure: 0.08
  max_drawdown: 0.08
  min_expected_value: 0.001

pricing:
  fd_solver: "rannacher"
  fd_grid_points: 300
  fd_time_steps: 300
  fd_s_max_multiple: 5.0
```

## Troubleshooting Examples

### 1. Handle Missing Data

```python
def handle_missing_data(market_data):
    """Handle missing data gracefully."""
    for key, data in market_data.items():
        if data is not None:
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # If still missing, backward fill
            data = data.fillna(method='bfill')
            
            # If still missing, interpolate
            data = data.interpolate()
            
            market_data[key] = data
    
    return market_data

# Use in strategy
market_data = handle_missing_data(market_data)
feature_set = feature_builder.build_features(**market_data)
```

### 2. Model Fallback

```python
def train_with_fallback(config, X, y):
    """Train model with fallback options."""
    models_to_try = ['lightgbm', 'catboost', 'sklearn']
    
    for model_type in models_to_try:
        try:
            config.model.model_type = model_type
            forecaster = GBTForecaster(config)
            results = forecaster.train(X, y)
            print(f"Successfully trained {model_type} model")
            return forecaster
        except Exception as e:
            print(f"Failed to train {model_type}: {e}")
            continue
    
    raise Exception("All models failed to train")

# Use fallback training
forecaster = train_with_fallback(config, X_train, y_train)
```

### 3. Performance Optimization

```python
import time
from functools import wraps

def time_execution(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Apply timing to key functions
@time_execution
def build_features_timed(feature_builder, market_data):
    return feature_builder.build_features(**market_data)

@time_execution
def train_model_timed(forecaster, X, y):
    return forecaster.train(X, y)

# Use timed functions
feature_set = build_features_timed(feature_builder, market_data)
model_results = train_model_timed(forecaster, feature_set.features, feature_set.target_direction)
```

These examples provide comprehensive coverage of the strategy framework, from basic usage to advanced customization and troubleshooting. Each example is designed to be practical and immediately usable in real trading scenarios.
