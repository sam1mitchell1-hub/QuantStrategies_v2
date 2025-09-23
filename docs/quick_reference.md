# Quick Reference Guide

This guide provides quick access to the most commonly used functions and classes in the QuantStrategies platform.

## Stochastic Processes

### Geometric Brownian Motion

```python
from stochastic import GeometricBrownianMotion

# Create and fit GBM
gbm = GeometricBrownianMotion()
fit_results = gbm.fit(data['close'])

# Simulate paths
paths = gbm.simulate_path(initial_value, time_steps)

# Get analytical values
mean = gbm.get_analytical_mean(initial_value, time)
variance = gbm.get_analytical_variance(initial_value, time)
ci_lower, ci_upper = gbm.get_confidence_intervals(initial_value, time)
```

### Key Parameters
- `mu`: Drift parameter (annualized)
- `sigma`: Volatility parameter (annualized)
- `method`: Fitting method ("mle" or "moments")

## PDE Solvers

### Black-Scholes Crank-Nicolson Solver

```python
from pde import BlackScholesCNSolver

# Create solver
solver = BlackScholesCNSolver(
    S_min=0.0,
    S_max=400.0,  # 4 × K
    T=1.0,        # Time to expiration
    r=0.05,       # Risk-free rate
    sigma=0.2,    # Volatility
    K=100.0,      # Strike price
    option_type='call',
    N_S=100,      # Spatial grid points
    N_T=100       # Time steps
)

# Solve
solver.setup_grid()
solution = solver.solve()

# Get results
price = solver.get_option_price(S=100.0, t=0.0)
greeks = solver.get_greeks(S=100.0, t=0.0)
```

### Key Parameters
- `S_min`, `S_max`: Price grid bounds
- `T`: Time to expiration
- `r`: Risk-free rate
- `sigma`: Volatility
- `K`: Strike price
- `option_type`: 'call' or 'put'
- `N_S`: Number of spatial grid points
- `N_T`: Number of time steps

## Data Management

### YFinance Client

```python
from quant_strategies.data import YFinanceClient

client = YFinanceClient()
data = client.get_daily_prices_from_start("AAPL", "2020-01-01")
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
pytest tests/test_stochastic_processes.py -v
pytest tests/test_pde_solvers.py -v
```

## Demo Scripts

### Stochastic Processes
```bash
python scripts/test_gbm.py
python scripts/test_gbm_synthetic.py
```

### PDE Solvers
```bash
python scripts/test_black_scholes_cn.py
```

## Common Patterns

### Monte Carlo Analysis
```python
# Generate multiple paths
n_paths = 1000
all_paths = np.zeros((n_paths, len(time_steps)))
for i in range(n_paths):
    all_paths[i] = gbm.simulate_path(initial_value, time_steps)

# Calculate statistics
final_values = all_paths[:, -1]
mean_final = np.mean(final_values)
std_final = np.std(final_values)
```

### Put-Call Parity Validation
```python
# Create call and put solvers
call_solver = BlackScholesCNSolver(option_type='call', K=100.0, T=1.0, r=0.05, sigma=0.2)
put_solver = BlackScholesCNSolver(option_type='put', K=100.0, T=1.0, r=0.05, sigma=0.2)

# Solve both
call_solver.setup_grid()
put_solver.setup_grid()
call_solver.solve()
put_solver.solve()

# Test parity: C - P = S - K*exp(-r*T)
S = 100.0
call_price = call_solver.get_option_price(S, t=0.0)
put_price = put_solver.get_option_price(S, t=0.0)
lhs = call_price - put_price
rhs = S - call_solver.K * np.exp(-call_solver.r * call_solver.T)
```

### Greeks Calculation
```python
# Get all Greeks
greeks = solver.get_greeks(S=100.0, t=0.0)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
```

## Error Handling

### Common Exceptions
- `ValueError`: Invalid parameters or insufficient data
- `RuntimeError`: Solver not initialized or grid not set up
- `ImportError`: Missing dependencies

### Best Practices
- Always call `setup_grid()` before `solve()`
- Check `solver.solved` before querying results
- Use appropriate grid sizes for accuracy vs. performance
- Validate input parameters before creating solvers
