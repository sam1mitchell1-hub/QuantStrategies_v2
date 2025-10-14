# Options Trading Strategy Framework

A comprehensive, production-ready options trading strategy that combines machine learning forecasting (P-measure) with finite difference option pricing (Q-measure) for systematic options spread trading.

## 🎯 Strategy Overview

This framework implements a complete options trading pipeline:

1. **Feature Engineering**: Builds comprehensive features from market data
2. **ML Forecasting**: Uses GBT models to predict market direction and volatility
3. **Structure Selection**: Selects appropriate bull/bear spreads based on forecasts
4. **FD Pricing**: Uses our Crank-Nicolson solvers for accurate option pricing
5. **Risk Management**: Implements Kelly sizing and comprehensive risk controls
6. **Backtesting**: Event-aligned backtesting with realistic execution simulation

## 🏗️ Architecture

```
strategy/
├── config.py          # Configuration management
├── features.py         # Feature engineering pipeline
├── forecasting.py      # GBT forecasting models
├── structures.py       # Option structure selection
├── pricing.py          # FD pricing integration
├── risk.py            # Risk management system
├── backtesting.py     # Backtesting framework
└── __init__.py        # Package initialization
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install numpy pandas matplotlib scipy scikit-learn

# ML libraries (choose one or both)
pip install lightgbm
# or
pip install catboost

# Optional: for advanced features
pip install yfinance pandas-datareader
```

### 2. Run the Strategy

```bash
python scripts/run_strategy.py
```

This will:
- Load the configuration from `config/strategy_config.yaml`
- Generate sample market data
- Build features and train the model
- Run a complete backtest
- Generate performance reports and plots

### 3. Customize Configuration

Edit `config/strategy_config.yaml` to adjust:
- Market parameters (underlyings, expiries, strikes)
- Model settings (LightGBM/CatBoost parameters)
- Risk limits and position sizing
- Execution costs and slippage

## 📊 Key Features

### Feature Engineering
- **Volatility Features**: Realized vol, bipower variation, jump detection
- **IV Surface Features**: Level, skew, term structure, VRP
- **Cross-Asset Features**: VIX, rates, credit spreads
- **Regime Features**: Volatility regimes, trend indicators
- **Liquidity Features**: Volume analysis, bid-ask spreads

### Forecasting Models
- **LightGBM/CatBoost**: Gradient boosting for classification/regression
- **Cross-Validation**: Purged time series splits to avoid look-ahead bias
- **Calibration**: Platt scaling or isotonic regression for probability calibration
- **Feature Importance**: Automatic feature selection and analysis

### Option Structures
- **Bull Call Spreads**: For bullish forecasts
- **Bear Put Spreads**: For bearish forecasts
- **Strike Selection**: Based on forecast bands and liquidity
- **Liquidity Filtering**: Ensures tradeable strikes

### FD Pricing Integration
- **Crank-Nicolson Solver**: High-accuracy option pricing
- **Rannacher Smoothing**: Enhanced accuracy near expiry
- **Greeks Calculation**: Delta, gamma, theta, vega, rho
- **IV Surface Fitting**: SVI or spline-based interpolation

### Risk Management
- **Kelly Sizing**: Optimal position sizing based on expected value
- **Risk Limits**: Delta, gamma, theta, vega exposure limits
- **Drawdown Control**: Maximum drawdown and daily loss limits
- **Real-time Monitoring**: Continuous risk assessment and alerts

### Backtesting
- **Event Alignment**: Proper timestamp alignment to avoid look-ahead bias
- **Realistic Execution**: Includes commissions, slippage, and market impact
- **Performance Metrics**: Sharpe ratio, hit rate, profit factor, VaR
- **Risk Analysis**: Drawdown analysis, stress testing

## 📈 Performance Metrics

The framework tracks comprehensive performance metrics:

### Trading Metrics
- Total trades, hit rate, win/loss ratio
- Total P&L, returns, Sharpe ratio
- Profit factor, average win/loss
- Commission and slippage costs

### Risk Metrics
- Maximum drawdown, VaR (95%, 99%)
- Portfolio volatility, expected shortfall
- Greeks exposure, concentration risk

### Trade Analysis
- Performance by structure type
- Exit reason analysis
- Feature importance ranking
- Regime-based performance

## 🔧 Configuration

The strategy is highly configurable through YAML:

```yaml
# Market Configuration
market:
  trading_horizon_days: 5
  decision_time: "15:30:00"
  underlyings: ["SPY", "QQQ", "IWM"]
  min_days_to_expiry: 3
  max_days_to_expiry: 30

# Model Configuration
model:
  model_type: "lightgbm"
  objective: "classification"
  lightgbm_params:
    num_leaves: 31
    learning_rate: 0.05
    # ... more parameters

# Risk Configuration
risk:
  max_position_size_pct: 0.01
  kelly_fraction: 0.25
  max_delta_exposure: 0.1
  max_drawdown: 0.05
```

## 📁 Output Files

The strategy generates comprehensive outputs:

- `output/strategy_performance.png`: Performance visualization
- `output/trade_summary.csv`: Detailed trade log
- `output/performance_metrics.csv`: Performance statistics
- `output/feature_importance.csv`: Feature ranking

## 🧪 Testing

The framework includes comprehensive testing:

```python
# Test individual components
from strategy import StrategyConfig, FeatureBuilder, GBTForecaster

config = StrategyConfig()
feature_builder = FeatureBuilder(config)
forecaster = GBTForecaster(config)

# Test with sample data
# ... (see run_strategy.py for examples)
```

## 🔮 Future Enhancements

- **Additional Models**: XGBoost, neural networks, ensemble methods
- **More Structures**: Iron condors, butterflies, straddles
- **Advanced Features**: Alternative data, sentiment analysis
- **Real-time Trading**: Live market data integration
- **Portfolio Optimization**: Multi-asset portfolio management

## 📚 References

This framework implements concepts from:
- Rannacher smoothing for FD methods
- Kelly criterion for position sizing
- Purged cross-validation for time series
- SVI model for IV surface fitting
- Modern portfolio theory and risk management

## 🤝 Contributing

The framework is designed to be modular and extensible. Key extension points:
- New feature types in `FeatureBuilder`
- Additional models in `GBTForecaster`
- New option structures in `OptionStructureSelector`
- Alternative pricing methods in `FDPricer`
- Enhanced risk metrics in `RiskManager`

## 📄 License

This project is part of the QuantStrategies framework and follows the same licensing terms.
