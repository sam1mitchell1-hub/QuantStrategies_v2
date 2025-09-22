#!/usr/bin/env python3
"""
Test script for the Geometric Brownian Motion implementation.

This script demonstrates how to fit a GBM to historical data and simulate
future paths for Monte Carlo analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from quant_strategies.data import YFinanceClient
from stochastic import GeometricBrownianMotion


def test_gbm_fitting():
    """Test GBM fitting to historical data."""
    print("Testing GBM fitting to historical data...")
    
    # Try to fetch some historical data
    client = YFinanceClient()
    data = client.get_daily_prices_from_start("AAPL", "2020-01-01")
    
    if data is None or len(data) < 100:
        print("Failed to fetch sufficient historical data - using synthetic data instead")
        # Generate synthetic data as fallback
        data = generate_synthetic_data(mu=0.08, sigma=0.25, initial_price=100, days=1000)
        print(f"Generated {len(data)} days of synthetic data")
    else:
        print(f"Fetched {len(data)} days of AAPL data")
    
    # Create and fit GBM
    gbm = GeometricBrownianMotion()
    fit_results = gbm.fit(data)
    
    print("\nFitting Results:")
    print(f"μ (drift): {float(fit_results['mu']):.6f}")
    print(f"σ (volatility): {float(fit_results['sigma']):.6f}")
    print(f"Log-likelihood: {float(fit_results['log_likelihood']):.2f}")
    print(f"Data points: {int(fit_results['data_points'])}")
    
    return gbm, data


def generate_synthetic_data(mu=0.1, sigma=0.2, initial_price=100, days=1000):
    """Generate synthetic GBM data for testing."""
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
    
    return pd.DataFrame({'close': prices}, index=dates)


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
    
    print(f"Initial value: ${float(initial_value):.2f}")
    print(f"Simulated mean final value: ${float(mean_final):.2f}")
    print(f"Analytical mean final value: ${float(analytical_mean):.2f}")
    print(f"Simulated std final value: ${float(std_final):.2f}")
    print(f"Analytical std final value: ${float(analytical_std):.2f}")
    
    # Confidence intervals
    ci_lower, ci_upper = gbm.get_confidence_intervals(initial_value, days/365.25)
    print(f"95% Confidence interval: [${float(ci_lower):.2f}, ${float(ci_upper):.2f}]")
    
    return paths, time_steps


def plot_results(data, gbm, paths, time_steps, initial_value):
    """Plot historical data and simulated paths."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Historical data
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['close'], 'b-', linewidth=1, label='Historical AAPL')
    plt.title('Historical AAPL Price Data')
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
    log_returns = np.diff(np.log(data['close']))
    plt.hist(log_returns, bins=50, alpha=0.7, density=True, label='Historical')
    
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
    plt.savefig('gbm_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main test function."""
    print("=== GBM Stochastic Process Test ===\n")
    
    try:
        # Test fitting
        gbm, data = test_gbm_fitting()
        
        if gbm is None:
            return
        
        # Test simulation
        initial_value = data['close'].iloc[-1]  # Use last price as starting point
        paths, time_steps = test_gbm_simulation(gbm, initial_value, days=252)
        
        # Plot results
        plot_results(data, gbm, paths, time_steps, initial_value)
        
        print("\n=== Test completed successfully! ===")
        print("Results saved to 'gbm_test_results.png'")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
