# Installation Guide

This guide shows how to install the QuantStrategies framework with different dependency sets.

## 🚀 Quick Installation

### Basic Installation (Core Framework)
```bash
pip install -e .
```

This installs the core framework with:
- PDE solvers (Crank-Nicolson, Rannacher)
- Stochastic processes (GBM)
- Basic data management
- Configuration system

### With ML Dependencies (Strategy Framework)
```bash
pip install -e .[strategy]
```

This adds:
- LightGBM
- CatBoost
- Scikit-learn
- PyYAML

### With All ML Libraries
```bash
pip install -e .[ml]
```

This includes:
- LightGBM
- CatBoost
- XGBoost
- Scikit-learn

### Development Setup
```bash
pip install -e .[dev]
```

This adds development tools:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)
- jupyter (notebooks)
- matplotlib (plotting)

### Complete Installation (Everything)
```bash
pip install -e .[strategy,ml,dev,notebook]
```

## 📦 Dependency Groups

| Group | Description | Key Dependencies |
|-------|-------------|------------------|
| `strategy` | Options trading strategy | lightgbm, catboost, scikit-learn |
| `ml` | All ML libraries | lightgbm, catboost, xgboost, scikit-learn |
| `dev` | Development tools | pytest, black, flake8, mypy |
| `notebook` | Jupyter notebooks | jupyter, plotly, matplotlib |
| `test` | Testing framework | pytest, matplotlib |

## 🔧 Using UV (Recommended)

If you're using UV (the fast Python package installer):

```bash
# Install with strategy dependencies
uv pip install -e .[strategy]

# Install everything
uv pip install -e .[strategy,ml,dev,notebook]
```

## 🐍 Python Version Requirements

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10 or 3.11
- **Strategy Framework**: Python 3.9+ (for ML libraries)

## 📋 Verification

After installation, verify everything works:

```python
# Test core framework
from pde import BlackScholesCNSolver
from stochastic import GeometricBrownianMotion

# Test strategy framework (if installed with [strategy])
from strategy import StrategyConfig, FeatureBuilder, GBTForecaster

print("✅ All components working!")
```

## 🚨 Troubleshooting

### LightGBM Installation Issues
```bash
# On macOS with M1/M2
brew install libomp
pip install lightgbm

# On Windows
pip install lightgbm --only-binary=all
```

### CatBoost Installation Issues
```bash
# Install from conda-forge
conda install -c conda-forge catboost

# Or use pip
pip install catboost
```

### Memory Issues with ML Libraries
```bash
# Install without OpenMP (reduces memory usage)
pip install lightgbm --no-deps
pip install scikit-learn numpy scipy
```

## 🔄 Updating Dependencies

To update all dependencies:

```bash
pip install -e .[strategy,ml,dev,notebook] --upgrade
```

## 📚 Next Steps

1. **Install with strategy dependencies**: `pip install -e .[strategy]`
2. **Run the strategy**: `python scripts/run_strategy.py`
3. **Explore examples**: Check the `scripts/` directory
4. **Read documentation**: See `strategy/README.md`

## 🤝 Contributing

For development, install with all dependencies:

```bash
pip install -e .[dev,strategy,ml,notebook]
```

This gives you everything needed for development, testing, and running the complete strategy framework.
