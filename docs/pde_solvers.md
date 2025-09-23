# PDE Solvers Framework

The PDE solvers framework provides finite difference methods for solving partial differential equations used in quantitative finance, particularly for options pricing models.

## Overview

The framework is designed to support:
- Black-Scholes equation solving using finite difference methods
- Crank-Nicolson scheme for stability and accuracy
- Log-space transformation for improved numerical properties
- European options pricing and Greeks calculation
- Extensible architecture for additional PDEs and solution methods

## Architecture

### Base Classes

#### `PDESolver`
Abstract base class for all PDE solvers defining the common interface:
- `setup_grid()`: Set up the computational grid
- `apply_boundary_conditions()`: Apply boundary conditions
- `apply_initial_conditions()`: Apply initial/terminal conditions
- `solve()`: Solve the PDE
- `get_solution_at_point()`: Get solution at specific coordinates

#### `BlackScholesSolver`
Specialized base class for Black-Scholes equation solvers with common functionality:
- Parameter validation
- Payoff function calculation
- Boundary condition handling
- Solution interpolation

### Implemented Solvers

#### Black-Scholes Crank-Nicolson Solver

The `BlackScholesCNSolver` implements the Black-Scholes PDE using the Crank-Nicolson scheme in log-space:

**PDE in log-space (x = ln(S)):**
```
∂V/∂t + ½σ²(∂²V/∂x²) + (r - ½σ²)(∂V/∂x) - rV = 0
```

**Features:**
- Crank-Nicolson scheme for stability and accuracy
- Log-space transformation for constant coefficients
- Thomas algorithm for efficient tridiagonal matrix solving
- Support for European call and put options
- Greeks calculation using finite differences

## Usage Examples

### Basic Usage

```python
from pde import BlackScholesCNSolver

# Create solver with parameters
solver = BlackScholesCNSolver(
    S_min=0.0,
    S_max=400.0,  # 4 × K
    T=1.0,        # 1 year
    r=0.05,       # 5% risk-free rate
    sigma=0.2,    # 20% volatility
    K=100.0,      # Strike price
    option_type='call',
    N_S=100,      # Spatial grid points
    N_T=100       # Time steps
)

# Setup and solve
solver.setup_grid()
solution = solver.solve()

# Get option price
price = solver.get_option_price(S=100.0, t=0.0)
```

### Greeks Calculation

```python
# Calculate option Greeks
greeks = solver.get_greeks(S=100.0, t=0.0)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
```

### Solution Surface Analysis

```python
# Get complete solution surface
surface = solver.get_solution_surface()
S_grid = surface['S_grid']
t_grid = surface['t_grid']
solution = surface['solution']

# Plot or analyze the solution surface
import matplotlib.pyplot as plt
plt.contourf(S_grid, t_grid, solution)
plt.xlabel('Underlying Price (S)')
plt.ylabel('Time to Expiration')
plt.title('Option Price Surface')
plt.colorbar()
plt.show()
```

### Put-Call Parity Validation

```python
# Create call and put solvers with same parameters
call_solver = BlackScholesCNSolver(option_type='call', K=100.0, T=1.0, r=0.05, sigma=0.2)
put_solver = BlackScholesCNSolver(option_type='put', K=100.0, T=1.0, r=0.05, sigma=0.2)

call_solver.setup_grid()
put_solver.setup_grid()
call_solver.solve()
put_solver.solve()

# Test put-call parity: C - P = S - K*exp(-r*T)
S = 100.0
call_price = call_solver.get_option_price(S, t=0.0)
put_price = put_solver.get_option_price(S, t=0.0)

lhs = call_price - put_price
rhs = S - call_solver.K * np.exp(-call_solver.r * call_solver.T)

print(f"Put-Call Parity: {lhs:.6f} ≈ {rhs:.6f}")
```

## API Reference

### BlackScholesCNSolver

#### Constructor
```python
BlackScholesCNSolver(
    S_min=0.0,
    S_max=None,  # Default: 4 * K
    T=1.0,
    r=0.05,
    sigma=0.2,
    K=100.0,
    option_type='call',
    N_S=100,
    N_T=100
)
```

**Parameters:**
- `S_min` (float): Minimum underlying price (default: 0.0)
- `S_max` (float): Maximum underlying price (default: 4 * K)
- `T` (float): Time to expiration (default: 1.0)
- `r` (float): Risk-free interest rate (default: 0.05)
- `sigma` (float): Volatility (default: 0.2)
- `K` (float): Strike price (default: 100.0)
- `option_type` (str): 'call' or 'put' (default: 'call')
- `N_S` (int): Number of spatial grid points (default: 100)
- `N_T` (int): Number of time steps (default: 100)

#### Methods

##### `setup_grid()`
Set up the computational grid in log-space.

##### `solve()`
Solve the Black-Scholes PDE using Crank-Nicolson scheme.

**Returns:**
- Solution array (time × space)

##### `get_option_price(S, t=0.0)`
Get option price at a specific underlying price and time.

**Parameters:**
- `S` (float): Underlying price
- `t` (float): Time (default: 0.0 for current time)

**Returns:**
- Option price

##### `get_greeks(S, t=0.0)`
Calculate option Greeks using finite differences.

**Parameters:**
- `S` (float): Underlying price
- `t` (float): Time (default: 0.0 for current time)

**Returns:**
- Dictionary containing Delta, Gamma, Theta

##### `get_solution_surface()`
Get the complete solution surface.

**Returns:**
- Dictionary containing S_grid, t_grid, and solution array

## Numerical Methods

### Crank-Nicolson Scheme

The Crank-Nicolson scheme provides a good balance between stability and accuracy:

- **Stability**: Unconditionally stable for all time steps
- **Accuracy**: Second-order accurate in both space and time
- **Implementation**: Implicit scheme requiring matrix solving

### Log-Space Transformation

Transforming to log-space (x = ln(S)) provides several advantages:

- **Constant coefficients**: Eliminates S-dependence in the PDE
- **Better numerical properties**: More uniform grid spacing
- **Improved stability**: Reduces numerical errors

### Thomas Algorithm

The Thomas algorithm efficiently solves tridiagonal systems:

- **Complexity**: O(n) for n equations
- **Memory efficient**: No need to store full matrix
- **Numerically stable**: Suitable for most financial PDEs

## Grid Design

### Spatial Grid
- **Log-space**: x = ln(S) for constant coefficients
- **Uniform spacing**: dx = (x_max - x_min) / (N_S - 1)
- **Boundaries**: S_min = 0, S_max = 4 × K (configurable)

### Time Grid
- **Uniform spacing**: dt = T / N_T
- **Backward stepping**: From expiration (t=T) to present (t=0)
- **Terminal condition**: Payoff function at expiration

### Boundary Conditions

#### At S = 0 (x → -∞)
- **Call option**: V(0, t) = 0
- **Put option**: V(0, t) = K × exp(-r(T-t))

#### At S = S_max (x → +∞)
- **Call option**: V(S_max, t) ≈ S_max - K × exp(-r(T-t))
- **Put option**: V(S_max, t) = 0

## Testing

The framework includes comprehensive unit tests covering:
- Parameter validation
- Grid setup and solving
- Option pricing accuracy
- Greeks calculation
- Put-call parity
- Thomas algorithm implementation

Run tests with:
```bash
pytest tests/test_pde_solvers.py -v
```

## Demo Scripts

Several demo scripts are available in the `scripts/` directory:
- `test_black_scholes_cn.py`: Comprehensive test with analytical comparison

## Future Extensions

The framework is designed for easy extension:

### Additional Solvers
- **Explicit schemes**: For simple cases and prototyping
- **Implicit schemes**: For maximum stability
- **ADI methods**: For higher-dimensional problems

### Additional Options
- **American options**: Early exercise boundary
- **Barrier options**: Knock-in/knock-out conditions
- **Asian options**: Path-dependent payoffs

### Additional PDEs
- **Heston model**: Stochastic volatility
- **Jump-diffusion**: Merton's model
- **Local volatility**: Dupire's equation

## Performance Considerations

### Grid Resolution
- **Spatial points**: More points improve accuracy but increase computation
- **Time steps**: More steps improve accuracy but increase computation
- **Typical values**: N_S = 100-500, N_T = 100-1000

### Memory Usage
- **Solution array**: N_S × N_T floating point numbers
- **Tridiagonal system**: 3 × N_S coefficients per time step
- **Typical memory**: ~1-10 MB for standard grids

### Computational Complexity
- **Per time step**: O(N_S) for Thomas algorithm
- **Total complexity**: O(N_S × N_T)
- **Typical runtime**: <1 second for standard grids
