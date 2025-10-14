#!/usr/bin/env python3
"""
Comparison script for standard Crank-Nicolson vs Crank-Nicolson with Rannacher smoothing.

This script runs both solvers on the same underlying parameters and compares:
- Solution accuracy
- Error analysis
- Greeks calculation
- Performance metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
from pde import BlackScholesCNSolver, BlackScholesCNRannacherSolver
from scipy.stats import norm

def analytical_black_scholes(S, K, T, r, sigma, option_type='call'):
    """Analytical Black-Scholes formula."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def run_solver_comparison(solver_params, rannacher_steps=4):
    """Run comparison between standard CN and Rannacher CN solvers."""
    print("=== CRANK-NICOLSON SOLVER COMPARISON ===\n")
    
    # Create both solvers
    print("Creating solvers...")
    cn_solver = BlackScholesCNSolver(**solver_params)
    rannacher_solver = BlackScholesCNRannacherSolver(**solver_params, rannacher_steps=rannacher_steps)
    
    # Setup grids
    print("Setting up grids...")
    cn_solver.setup_grid()
    rannacher_solver.setup_grid()
    
    # Apply initial conditions
    cn_solver.apply_initial_conditions()
    rannacher_solver.apply_initial_conditions()
    
    # Solve with timing
    print("Solving with standard Crank-Nicolson...")
    start_time = time.time()
    cn_solver.solve()
    cn_time = time.time() - start_time
    
    print("Solving with Rannacher smoothing...")
    start_time = time.time()
    rannacher_solver.solve()
    rannacher_time = time.time() - start_time
    
    print(f"Standard CN time: {cn_time:.4f} seconds")
    print(f"Rannacher CN time: {rannacher_time:.4f} seconds")
    print()
    
    return cn_solver, rannacher_solver, cn_time, rannacher_time

def compare_solutions(cn_solver, rannacher_solver, test_points):
    """Compare solution accuracy between the two solvers."""
    print("=== SOLUTION ACCURACY COMPARISON ===")
    print(f"{'S':>8} {'Analytical':>12} {'Standard CN':>12} {'Rannacher':>12} {'CN Error':>12} {'Rannacher Error':>15}")
    print("-" * 80)
    
    cn_errors = []
    rannacher_errors = []
    
    for S in test_points:
        # Analytical solution
        analytical = analytical_black_scholes(S, cn_solver.K, cn_solver.T, cn_solver.r, cn_solver.sigma, cn_solver.option_type)
        
        # Finite difference solutions
        cn_price = cn_solver.get_option_price(S, t=0.0)
        rannacher_price = rannacher_solver.get_option_price(S, t=0.0)
        
        # Errors
        cn_error = cn_price - analytical
        rannacher_error = rannacher_price - analytical
        
        cn_errors.append(cn_error)
        rannacher_errors.append(rannacher_error)
        
        print(f"{S:>8.1f} {analytical:>12.6f} {cn_price:>12.6f} {rannacher_price:>12.6f} {cn_error:>12.6f} {rannacher_error:>15.6f}")
    
    # Summary statistics
    cn_rmse = np.sqrt(np.mean(np.array(cn_errors)**2))
    rannacher_rmse = np.sqrt(np.mean(np.array(rannacher_errors)**2))
    
    print("-" * 80)
    print(f"{'RMSE':>8} {'':>12} {'':>12} {'':>12} {cn_rmse:>12.6f} {rannacher_rmse:>15.6f}")
    print()
    
    return cn_errors, rannacher_errors, cn_rmse, rannacher_rmse

def compare_greeks(cn_solver, rannacher_solver, S_test):
    """Compare Greeks calculation between the two solvers."""
    print("=== GREEKS COMPARISON ===")
    print(f"Testing at S = {S_test}")
    print()
    
    # Get Greeks from both solvers
    cn_greeks = cn_solver.get_greeks(S_test, t=0.0)
    rannacher_greeks = rannacher_solver.get_greeks(S_test, t=0.0)
    
    print(f"{'Greek':>8} {'Standard CN':>15} {'Rannacher':>15} {'Difference':>15}")
    print("-" * 60)
    
    for greek in ['delta', 'gamma', 'theta']:
        cn_val = cn_greeks[greek]
        rannacher_val = rannacher_greeks[greek]
        difference = rannacher_val - cn_val
        
        print(f"{greek:>8} {cn_val:>15.6f} {rannacher_val:>15.6f} {difference:>15.6f}")
    
    print()
    return cn_greeks, rannacher_greeks

def create_comparison_plots(cn_solver, rannacher_solver, test_points, cn_errors, rannacher_errors, cn_time, rannacher_time):
    """Create comprehensive comparison plots."""
    print("=== CREATING COMPARISON PLOTS ===")
    
    # Get solution surfaces
    cn_surface = cn_solver.get_solution_surface()
    rannacher_surface = rannacher_solver.get_solution_surface()
    
    # Create figure with subplots
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Solution comparison
    plt.subplot(2, 3, 1)
    S_test_range = np.linspace(80, 120, 100)
    cn_prices = [cn_solver.get_option_price(S, t=0.0) for S in S_test_range]
    rannacher_prices = [rannacher_solver.get_option_price(S, t=0.0) for S in S_test_range]
    analytical_prices = [analytical_black_scholes(S, cn_solver.K, cn_solver.T, cn_solver.r, cn_solver.sigma, cn_solver.option_type) for S in S_test_range]
    
    plt.plot(S_test_range, analytical_prices, 'k-', linewidth=2, label='Analytical')
    plt.plot(S_test_range, cn_prices, 'b--', linewidth=2, label='Standard CN')
    plt.plot(S_test_range, rannacher_prices, 'r:', linewidth=2, label='Rannacher CN')
    plt.axvline(x=cn_solver.K, color='g', linestyle=':', alpha=0.7, label=f'Strike K={cn_solver.K}')
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Option Price')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    plt.subplot(2, 3, 2)
    cn_errors_range = np.array(cn_prices) - np.array(analytical_prices)
    rannacher_errors_range = np.array(rannacher_prices) - np.array(analytical_prices)
    
    plt.plot(S_test_range, cn_errors_range, 'b--', linewidth=2, label='Standard CN Error')
    plt.plot(S_test_range, rannacher_errors_range, 'r:', linewidth=2, label='Rannacher CN Error')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=cn_solver.K, color='g', linestyle=':', alpha=0.7)
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Error')
    plt.title('Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Solution surfaces comparison
    plt.subplot(2, 3, 3)
    # Show difference between solutions
    solution_diff = rannacher_solver.solution - cn_solver.solution
    S_mesh, T_mesh = np.meshgrid(cn_solver.S_grid, cn_solver.t_grid, indexing='ij')
    
    contour = plt.contourf(S_mesh, T_mesh, solution_diff.T, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, label='Rannacher - Standard CN')
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Time to Expiration')
    plt.title('Solution Difference Surface')
    
    # Plot 4: Error evolution over time
    plt.subplot(2, 3, 4)
    time_indices = [0, cn_solver.N_T//4, cn_solver.N_T//2, 3*cn_solver.N_T//4, cn_solver.N_T]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    
    for i, color in zip(time_indices, colors):
        time_to_expiry = cn_solver.T - cn_solver.t_grid[i]
        cn_sol = cn_solver.solution[i, :]
        rannacher_sol = rannacher_solver.solution[i, :]
        diff = rannacher_sol - cn_sol
        
        plt.plot(cn_solver.S_grid, diff, color=color, linewidth=1.5, 
                label=f't = {cn_solver.t_grid[i]:.2f} (T-t = {time_to_expiry:.2f})')
    
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Rannacher - Standard CN')
    plt.title('Solution Difference Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Rannacher smoothing info
    plt.subplot(2, 3, 5)
    rannacher_info = rannacher_solver.get_rannacher_info()
    
    # Show which time steps use which scheme
    time_steps = np.arange(cn_solver.N_T + 1)
    scheme_type = np.zeros(cn_solver.N_T + 1)
    
    for i in range(cn_solver.N_T + 1):
        if i >= cn_solver.N_T - rannacher_info['rannacher_steps']:
            scheme_type[i] = 1  # Rannacher
        else:
            scheme_type[i] = 0  # Crank-Nicolson
    
    plt.plot(cn_solver.t_grid, scheme_type, 'b-', linewidth=3, label='Scheme Type')
    plt.fill_between(cn_solver.t_grid, 0, scheme_type, alpha=0.3, color='blue', label='Rannacher Steps')
    plt.fill_between(cn_solver.t_grid, scheme_type, 1, alpha=0.3, color='red', label='CN Steps')
    plt.xlabel('Time')
    plt.ylabel('Scheme (0=CN, 1=Rannacher)')
    plt.title('Rannacher Smoothing Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Performance comparison
    plt.subplot(2, 3, 6)
    methods = ['Standard CN', 'Rannacher CN']
    times = [cn_time, rannacher_time]
    colors = ['blue', 'red']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison')
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/cn_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plots saved to 'output/cn_solver_comparison.png'")

def main():
    """Main comparison function."""
    print("=== CRANK-NICOLSON SOLVER COMPARISON ===\n")
    
    # Solver parameters
    solver_params = {
        'S_min': 0.0,
        'S_max': 200.0,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.2,
        'K': 100.0,
        'option_type': 'call',
        'N_S': 200,
        'N_T': 200
    }
    
    # Rannacher smoothing parameters
    rannacher_steps = 4
    
    # Test points
    test_points = [80, 90, 100, 110, 120]
    S_test = 100.0  # For Greeks comparison
    
    print(f"Solver parameters: {solver_params}")
    print(f"Rannacher steps: {rannacher_steps}")
    print()
    
    # Run comparison
    cn_solver, rannacher_solver, cn_time, rannacher_time = run_solver_comparison(solver_params, rannacher_steps)
    
    # Compare solutions
    cn_errors, rannacher_errors, cn_rmse, rannacher_rmse = compare_solutions(cn_solver, rannacher_solver, test_points)
    
    # Compare Greeks
    cn_greeks, rannacher_greeks = compare_greeks(cn_solver, rannacher_solver, S_test)
    
    # Create plots
    create_comparison_plots(cn_solver, rannacher_solver, test_points, cn_errors, rannacher_errors, cn_time, rannacher_time)
    
    # Summary
    print("=== SUMMARY ===")
    print(f"Standard CN RMSE: {cn_rmse:.6f}")
    print(f"Rannacher CN RMSE: {rannacher_rmse:.6f}")
    print(f"Improvement: {((cn_rmse - rannacher_rmse) / cn_rmse) * 100:.2f}%")
    print(f"Standard CN time: {cn_time:.4f} seconds")
    print(f"Rannacher CN time: {rannacher_time:.4f} seconds")
    print(f"Time overhead: {((rannacher_time - cn_time) / cn_time) * 100:.2f}%")
    
    print("\n=== RANNACHER SMOOTHING INFO ===")
    rannacher_info = rannacher_solver.get_rannacher_info()
    print(f"Rannacher steps: {rannacher_info['rannacher_steps']}")
    print(f"Rannacher time steps: {rannacher_info['rannacher_time_steps']}")
    print(f"CN time steps: {rannacher_info['cn_time_steps']}")
    print(f"Rannacher θ: {rannacher_info['theta_rannacher']}")
    print(f"CN θ: {rannacher_info['theta_cn']}")

if __name__ == "__main__":
    main()
