#!/usr/bin/env python3
"""
Test script for the Black-Scholes Crank-Nicolson solver.

This script demonstrates how to use the PDE solver with dummy data and
compares results with analytical Black-Scholes formulas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pde import BlackScholesCNSolver


def analytical_black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate analytical Black-Scholes option price.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price


def test_basic_functionality(solver_params):
    """Test basic solver functionality with given parameters."""
    print("=== Testing Basic Functionality ===")
    
    # Create solver with provided parameters
    solver = BlackScholesCNSolver(**solver_params)
    
    # Setup and solve
    solver.setup_grid()
    solution = solver.solve()
    
    print(f"Solver: {solver.name}")
    print(f"Grid size: {solver.N_S} x {solver.N_T}")
    print(f"Solution shape: {solution.shape}")
    print(f"Solved: {solver.solved}")
    
    return solver


def test_option_pricing(solver, test_points=None):
    """Test option pricing at various underlying prices."""
    print("\n=== Testing Option Pricing ===")
    
    if test_points is None:
        test_points = [80, 90, 100, 110, 120]
    
    print(f"{'S':>8} {'FD Price':>12} {'Analytical':>12} {'Error':>10} {'Error %':>10}")
    print("-" * 60)
    
    for S in test_points:
        # Finite difference price
        fd_price = solver.get_option_price(S, t=0.0)
        
        # Analytical price
        analytical_price = analytical_black_scholes(
            S, solver.K, solver.T, solver.r, solver.sigma, solver.option_type
        )
        
        # Calculate error
        error = abs(fd_price - analytical_price)
        error_pct = (error / analytical_price) * 100 if analytical_price > 0 else 0
        
        print(f"{S:8.1f} {fd_price:12.6f} {analytical_price:12.6f} {error:10.6f} {error_pct:10.2f}%")


def test_greeks(solver, S_test=100.0):
    """Test Greeks calculation."""
    print(f"\n=== Testing Greeks Calculation ===")
    
    greeks = solver.get_greeks(S_test, t=0.0)
    
    print(f"Greeks at S = {S_test}:")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")


def test_put_call_parity(solver_params, S_test=100.0):
    """Test put-call parity relationship."""
    print("\n=== Testing Put-Call Parity ===")
    
    # Create call and put solvers with same parameters
    call_params = solver_params.copy()
    call_params['option_type'] = 'call'
    call_solver = BlackScholesCNSolver(**call_params)
    
    put_params = solver_params.copy()
    put_params['option_type'] = 'put'
    put_solver = BlackScholesCNSolver(**put_params)
    
    # Setup and solve both
    call_solver.setup_grid()
    put_solver.setup_grid()
    call_solver.solve()
    put_solver.solve()
    
    # Test put-call parity: C - P = S - K*exp(-r*T)
    call_price = call_solver.get_option_price(S_test, t=0.0)
    put_price = put_solver.get_option_price(S_test, t=0.0)
    
    lhs = call_price - put_price
    rhs = S_test - call_solver.K * np.exp(-call_solver.r * call_solver.T)
    
    print(f"Put-Call Parity Test:")
    print(f"  Call price: {call_price:.6f}")
    print(f"  Put price: {put_price:.6f}")
    print(f"  C - P: {lhs:.6f}")
    print(f"  S - K*exp(-r*T): {rhs:.6f}")
    print(f"  Difference: {abs(lhs - rhs):.6f}")
    
    return call_solver, put_solver


def plot_solution_surface(solver, save_plot=True):
    """Plot the solution surface."""
    print("\n=== Creating Solution Surface Plot ===")
    
    # Get solution surface
    surface = solver.get_solution_surface()
    S_grid = surface['S_grid']
    t_grid = surface['t_grid']
    solution = surface['solution']
    
    # Create meshgrid for plotting
    T_mesh, S_mesh = np.meshgrid(t_grid, S_grid, indexing='ij')
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # 3D surface plot
    ax = plt.subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(S_mesh, T_mesh, solution, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Underlying Price (S)')
    ax.set_ylabel('Time to Expiration')
    ax.set_zlabel('Option Price')
    ax.set_title(f'{solver.option_type.capitalize()} Option Price Surface')
    
    # 2D cross-sections
    plt.subplot(2, 2, 2)
    time_indices = [0, solver.N_T//4, solver.N_T//2, 3*solver.N_T//4, solver.N_T]
    for i in time_indices:
        plt.plot(S_grid, solution[i, :], label=f't = {t_grid[i]:.2f}')
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Option Price')
    plt.title('Price Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Payoff comparison
    plt.subplot(2, 2, 3)
    payoff = solver.get_payoff(S_grid)
    plt.plot(S_grid, payoff, 'k--', label='Payoff', linewidth=2)
    plt.plot(S_grid, solution[0, :], 'r-', label='FD Solution (t=0)', linewidth=2)
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Option Price')
    plt.title('Payoff vs FD Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error analysis
    plt.subplot(2, 2, 4)
    S_test = np.linspace(80, 120, 20)
    fd_prices = [solver.get_option_price(S, t=0.0) for S in S_test]
    analytical_prices = [analytical_black_scholes(S, solver.K, solver.T, solver.r, solver.sigma, solver.option_type) for S in S_test]
    
    errors = np.array(fd_prices) - np.array(analytical_prices)
    plt.plot(S_test, errors, 'bo-', markersize=4)
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('FD Error')
    plt.title('Finite Difference Error vs Analytical')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('output/black_scholes_cn_test_results.png', dpi=300, bbox_inches='tight')
        print("Results saved to 'output/black_scholes_cn_test_results.png'")
    
    plt.show()


def main():
    """Main test function with configurable parameters."""
    print("=== Black-Scholes Crank-Nicolson Solver Test ===\n")
    
    # =============================================================================
    # CONFIGURABLE PARAMETERS - Modify these to test different settings
    # =============================================================================
    
    # Solver parameters - modify these to experiment with different settings
    solver_params = {
        'S_min': 0.0,
        'S_max': 400.0,  # 4 × K
        'T': 1.0,        # Time to expiration
        'r': 0.05,       # Risk-free rate
        'sigma': 0.2,    # Volatility
        'K': 100.0,      # Strike price
        'option_type': 'call',
        'N_S': 100,      # Spatial grid points
        'N_T': 100       # Time steps
    }
    
    # Test points for option pricing comparison
    test_points = [80, 90, 100, 110, 120]
    
    # Point for Greeks calculation
    greeks_S = 100.0
    
    # Point for put-call parity test
    parity_S = 100.0
    
    # Whether to save plots
    save_plots = True
    
    # =============================================================================
    # END CONFIGURABLE PARAMETERS
    # =============================================================================
    
    try:
        # Test basic functionality
        solver = test_basic_functionality(solver_params)
        
        # Test option pricing
        test_option_pricing(solver, test_points)
        
        # Test Greeks
        test_greeks(solver, greeks_S)
        
        # Test put-call parity
        call_solver, put_solver = test_put_call_parity(solver_params, parity_S)
        
        # Create visualizations
        plot_solution_surface(solver, save_plots)
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()