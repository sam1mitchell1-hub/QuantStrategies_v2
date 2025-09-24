# Stochastic Processes Framework

The stochastic processes framework provides a comprehensive toolkit for modeling and simulating various stochastic processes used in quantitative finance, particularly for options pricing and risk management.

## Overview

The framework is designed to support:
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

## Usage Examples

### Basic Usage

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
```

### Monte Carlo Analysis

```python
# For Monte Carlo analysis
n_paths = 1000
all_paths = np.zeros((n_paths, len(time_steps)))
for i in range(n_paths):
    all_paths[i] = gbm.simulate_path(initial_value, time_steps)

# Calculate statistics
final_values = all_paths[:, -1]
mean_final = np.mean(final_values)
std_final = np.std(final_values)
```

### Parameter Fitting

```python
# Fit with different methods
gbm_mle = GeometricBrownianMotion()
gbm_moments = GeometricBrownianMotion()

# Maximum likelihood estimation
results_mle = gbm_mle.fit(data, method="mle")

# Method of moments
results_moments = gbm_moments.fit(data, method="moments")
```

### Analytical Calculations

```python
# Get analytical mean and variance
analytical_mean = gbm.get_analytical_mean(initial_value, time_horizon)
analytical_variance = gbm.get_analytical_variance(initial_value, time_horizon)

# Get confidence intervals
ci_lower, ci_upper = gbm.get_confidence_intervals(initial_value, time_horizon, confidence=0.95)
```

## API Reference

### GeometricBrownianMotion

#### Constructor
```python
GeometricBrownianMotion(mu=None, sigma=None)
```

**Parameters:**
- `mu` (float, optional): Drift parameter (if None, will be fitted from data)
- `sigma` (float, optional): Volatility parameter (if None, will be fitted from data)

#### Methods

##### `fit(data, method="mle")`
Fit GBM parameters to historical data.

**Parameters:**
- `data` (pandas.Series): Historical price data with datetime index
- `method` (str): Fitting method ("mle" or "moments")

**Returns:**
- Dictionary containing fitted parameters and fit statistics

##### `simulate_path(initial_value, time_steps, random_shocks=None)`
Simulate a complete path of the GBM process.

**Parameters:**
- `initial_value` (float): Starting value of the process
- `time_steps` (np.ndarray): Array of time points for the simulation
- `random_shocks` (np.ndarray, optional): Pre-generated random shocks

**Returns:**
- Array of simulated values

##### `get_analytical_mean(initial_value, time)`
Get the analytical expected value at time t.

**Parameters:**
- `initial_value` (float): Initial value of the process
- `time` (float): Time horizon (in years)

**Returns:**
- Expected value at time t

##### `get_analytical_variance(initial_value, time)`
Get the analytical variance at time t.

**Parameters:**
- `initial_value` (float): Initial value of the process
- `time` (float): Time horizon (in years)

**Returns:**
- Variance at time t

##### `get_confidence_intervals(initial_value, time, confidence=0.95)`
Get confidence intervals for the process at time t.

**Parameters:**
- `initial_value` (float): Initial value of the process
- `time` (float): Time horizon (in years)
- `confidence` (float): Confidence level (e.g., 0.95 for 95% CI)

**Returns:**
- Tuple of (lower_bound, upper_bound)

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

## Testing

The framework includes comprehensive unit tests covering:
- Parameter estimation accuracy
- Path simulation correctness
- Analytical method validation
- Error handling and edge cases

Run tests with:
```bash
pytest tests/test_stochastic_processes.py -v
```

## Demo Scripts

Several demo scripts are available in the `scripts/` directory:
- `test_gbm.py`: Test with historical data
- `test_gbm_synthetic.py`: Test with synthetic data
- `stochastic_processes_demo.py`: Jupyter notebook demo
