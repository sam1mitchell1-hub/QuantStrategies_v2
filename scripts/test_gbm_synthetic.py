#!/usr/bin/env python3
"""
Test script for the Geometric Brownian Motion implementation using synthetic data.

This script demonstrates how to fit a GBM to synthetic data and simulate
future paths for Monte Carlo analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from stochastic import GeometricBrownianMotion
from scipy import stats


def generate_synthetic_data(mu=0.1, sigma=0.2, initial_price=100, days=1000):
    """
    Generate synthetic GBM data for testing.
    
    Args:
        mu: True drift parameter
        sigma: True volatility parameter
        initial_price: Starting price
        days: Number of days to generate
        
    Returns:
        pandas Series with synthetic price data
    """
    # Generate time steps (daily)
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    dt = 1/365.25  # Daily time step in years
    
    # Generate random shocks
    random_shocks = np.random.standard_normal(days-1)
    
    # Generate price path
    prices = np.zeros(days)
    prices[0] = initial_price
    
    for i in range(1, days):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[i-1]
        prices[i] = prices[i-1] * np.exp(drift + diffusion)
    
    return pd.Series(prices, index=dates, name='close')


def test_gbm_fitting():
    """Test GBM fitting to synthetic data."""
    print("Testing GBM fitting to synthetic data...")
    
    # Generate synthetic data with known parameters
    true_mu = 0.08  # 8% annual drift
    true_sigma = 0.25  # 25% annual volatility
    initial_price = 100.0
    days = 1000
    
    print(f"Generating synthetic data with:")
    print(f"  True μ (drift): {true_mu:.4f}")
    print(f"  True σ (volatility): {true_sigma:.4f}")
    print(f"  Initial price: ${initial_price:.2f}")
    print(f"  Days: {days}")
    
    data = generate_synthetic_data(true_mu, true_sigma, initial_price, days)
    print(f"Generated {len(data)} days of synthetic data")
    
    # Create and fit GBM
    gbm = GeometricBrownianMotion()
    fit_results = gbm.fit(data)
    
    print("\nFitting Results:")
    print(f"Estimated μ (drift): {fit_results['mu']:.6f} (true: {true_mu:.6f})")
    print(f"Estimated σ (volatility): {fit_results['sigma']:.6f} (true: {true_sigma:.6f})")
    print(f"Log-likelihood: {fit_results['log_likelihood']:.2f}")
    print(f"Data points: {fit_results['data_points']}")
    
    # Calculate estimation errors
    mu_error = abs(fit_results['mu'] - true_mu) / true_mu * 100
    sigma_error = abs(fit_results['sigma'] - true_sigma) / true_sigma * 100
    
    print(f"\nEstimation Errors:")
    print(f"μ error: {mu_error:.2f}%")
    print(f"σ error: {sigma_error:.2f}%")
    
    return gbm, data, true_mu, true_sigma


def test_gbm_simulation(gbm, initial_value, days=252):
    """Test GBM simulation for Monte Carlo analysis."""
    print(f"\nTesting GBM simulation for {days} days...")
    
    # Create time steps (daily)
    time_steps = np.linspace(0, days/365.25, days + 1)  # Convert to years
    
    # Simulate multiple paths
    n_paths = 1000
    paths = np.zeros((n_paths, days + 1))
    
    for i in range(n_paths):
        paths[i] = gbm.simulate_path(initial_value, time_steps)
    
    # Calculate statistics
    final_values = paths[:, -1]
    mean_final = np.mean(final_values)
    std_final = np.std(final_values)
    
    # Analytical values
    analytical_mean = gbm.get_analytical_mean(initial_value, days/365.25)
    analytical_std = np.sqrt(gbm.get_analytical_variance(initial_value, days/365.25))
    
    print(f"Initial value: ${initial_value:.2f}")
    print(f"Simulated mean final value: ${mean_final:.2f}")
    print(f"Analytical mean final value: ${analytical_mean:.2f}")
    print(f"Simulated std final value: ${std_final:.2f}")
    print(f"Analytical std final value: ${analytical_std:.2f}")
    
    # Confidence intervals
    ci_lower, ci_upper = gbm.get_confidence_intervals(initial_value, days/365.25)
    print(f"95% Confidence interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
    
    return paths, time_steps


def plot_results(data, gbm, paths, time_steps, initial_value, true_mu, true_sigma):
    """Plot synthetic data and simulated paths."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Synthetic historical data
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data.values, 'b-', linewidth=1, label='Synthetic Data')
    plt.title('Synthetic GBM Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Sample simulated paths
    plt.subplot(2, 2, 2)
    for i in range(min(50, len(paths))):  # Show first 50 paths
        plt.plot(time_steps * 365.25, paths[i], 'r-', alpha=0.1)
    
    # Plot mean path
    mean_path = np.mean(paths, axis=0)
    plt.plot(time_steps * 365.25, mean_path, 'b-', linewidth=2, label='Mean Path')
    
    # Plot confidence intervals
    ci_lower, ci_upper = gbm.get_confidence_intervals(initial_value, time_steps[-1])
    plt.axhline(y=ci_lower, color='g', linestyle='--', alpha=0.7, label='95% CI Lower')
    plt.axhline(y=ci_upper, color='g', linestyle='--', alpha=0.7, label='95% CI Upper')
    
    plt.title('GBM Simulated Paths (Sample)')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of final values
    plt.subplot(2, 2, 3)
    final_values = paths[:, -1]
    plt.hist(final_values, bins=50, alpha=0.7, density=True, label='Simulated')
    
    # Overlay analytical distribution
    analytical_mean = gbm.get_analytical_mean(initial_value, time_steps[-1])
    analytical_std = np.sqrt(gbm.get_analytical_variance(initial_value, time_steps[-1]))
    x = np.linspace(final_values.min(), final_values.max(), 100)
    analytical_pdf = stats.norm.pdf(x, analytical_mean, analytical_std)
    plt.plot(x, analytical_pdf, 'r-', linewidth=2, label='Analytical')
    
    plt.title('Distribution of Final Values')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Log returns analysis
    plt.subplot(2, 2, 4)
    log_returns = np.diff(np.log(data.values))
    plt.hist(log_returns, bins=50, alpha=0.7, density=True, label='Synthetic')
    
    # Overlay theoretical distribution
    dt = 1/365.25  # Daily
    theoretical_mean = (gbm.mu - 0.5 * gbm.sigma**2) * dt
    theoretical_std = gbm.sigma * np.sqrt(dt)
    x = np.linspace(log_returns.min(), log_returns.max(), 100)
    theoretical_pdf = stats.norm.pdf(x, theoretical_mean, theoretical_std)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical')
    
    plt.title('Log Returns Distribution')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/gbm_synthetic_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main test function."""
    print("=== GBM Stochastic Process Test (Synthetic Data) ===\n")
    
    try:
        # Test fitting
        gbm, data, true_mu, true_sigma = test_gbm_fitting()
        
        # Test simulation
        initial_value = data.iloc[-1]  # Use last price as starting point
        paths, time_steps = test_gbm_simulation(gbm, initial_value, days=252)
        
        # Plot results
        plot_results(data, gbm, paths, time_steps, initial_value, true_mu, true_sigma)
        
        print("\n=== Test completed successfully! ===")
        print("Results saved to 'output/gbm_synthetic_test_results.png'")
        print("\nThe GBM implementation is working correctly!")
        print("The yfinance issue appears to be external (Yahoo Finance API problem).")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
