# Stochastic Processes Module

This module provides a framework for implementing and simulating various stochastic processes used in quantitative finance, particularly for options pricing and risk management.

## Overview

The stochastic processes framework is designed to support:
- Fitting stochastic processes to historical data
- Simulating future paths for Monte Carlo analysis
- Delta hedging strategy evaluation
- Options pricing and risk management

## Architecture

### Base Class: `StochasticProcess`

All stochastic processes inherit from the abstract `StochasticProcess` base class, which defines the common interface:

- `fit(data, **kwargs)`: Fit the process to historical data
- `step(current_value, dt, random_shock)`: Step forward one time increment
- `simulate_path(initial_value, time_steps, random_shocks)`: Simulate complete paths
- `get_required_parameters()`: Get list of required parameters

### Implemented Processes

#### Geometric Brownian Motion (GBM)

The `GeometricBrownianMotion` class implements the classic GBM process:

```
dS = μS dt + σS dW
```

Where:
- S is the asset price
- μ is the drift (expected return)
- σ is the volatility
- dW is a Wiener process (Brownian motion)

**Features:**
- Maximum likelihood and method of moments fitting
- Analytical mean and variance calculations
- Confidence interval computation
- Monte Carlo path simulation

## Usage Example

```python
from stochastic import GeometricBrownianMotion
from quant_strategies.data import YFinanceClient

# Fetch historical data
client = YFinanceClient()
data = client.get_daily_prices_from_start("AAPL", "2020-01-01")

# Create and fit GBM
gbm = GeometricBrownianMotion()
fit_results = gbm.fit(data['close'])

# Simulate future paths
initial_value = data['close'].iloc[-1]
time_steps = np.linspace(0, 1, 253)  # One year, daily steps
paths = gbm.simulate_path(initial_value, time_steps)

# For Monte Carlo analysis
n_paths = 1000
all_paths = np.zeros((n_paths, len(time_steps)))
for i in range(n_paths):
    all_paths[i] = gbm.simulate_path(initial_value, time_steps)
```

## Future Extensions

This framework is designed to be extensible. Future implementations could include:

- **Heston Model**: Stochastic volatility
- **Jump Diffusion**: Merton's jump-diffusion model
- **Mean Reverting**: Ornstein-Uhlenbeck process
- **Regime Switching**: Markov-switching models
- **Levy Processes**: More general jump processes

## Delta Hedging Integration

The framework is specifically designed to support delta hedging Monte Carlo analysis:

1. **Path Simulation**: Generate multiple underlying price paths
2. **Delta Calculation**: At each time step, calculate option delta
3. **Hedge Rebalancing**: Simulate delta hedging strategy
4. **P&L Analysis**: Evaluate hedging performance across all paths

This enables comprehensive analysis of hedging strategies under various market conditions.
