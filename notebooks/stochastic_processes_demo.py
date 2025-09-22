"""
Stochastic Processes Demo

This notebook demonstrates how to use the stochastic processes framework
for fitting GBM to historical data and simulating future paths for
Monte Carlo analysis of delta hedging strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quant_strategies.data import YFinanceClient
from stochastic import GeometricBrownianMotion

# Fetch historical data
print("Fetching historical data...")
client = YFinanceClient()
data = client.get_daily_prices_from_start("AAPL", "2020-01-01")

if data is not None and len(data) > 100:
    print(f"Fetched {len(data)} days of AAPL data")
    
    # Create and fit GBM
    print("\nFitting GBM to historical data...")
    gbm = GeometricBrownianMotion()
    fit_results = gbm.fit(data['close'])
    
    print(f"Fitted parameters:")
    print(f"  μ (drift): {fit_results['mu']:.6f}")
    print(f"  σ (volatility): {fit_results['sigma']:.6f}")
    print(f"  Log-likelihood: {fit_results['log_likelihood']:.2f}")
    
    # Simulate future paths
    print("\nSimulating future paths...")
    initial_value = data['close'].iloc[-1]
    days = 252  # One year
    time_steps = np.linspace(0, days/365.25, days + 1)
    
    # Simulate multiple paths for Monte Carlo
    n_paths = 1000
    paths = np.zeros((n_paths, days + 1))
    
    for i in range(n_paths):
        paths[i] = gbm.simulate_path(initial_value, time_steps)
    
    # Calculate statistics
    final_values = paths[:, -1]
    mean_final = np.mean(final_values)
    std_final = np.std(final_values)
    
    print(f"Monte Carlo Results ({n_paths} paths):")
    print(f"  Initial value: ${initial_value:.2f}")
    print(f"  Mean final value: ${mean_final:.2f}")
    print(f"  Std final value: ${std_final:.2f}")
    
    # Analytical comparison
    analytical_mean = gbm.get_analytical_mean(initial_value, days/365.25)
    analytical_std = np.sqrt(gbm.get_analytical_variance(initial_value, days/365.25))
    
    print(f"\nAnalytical Results:")
    print(f"  Expected final value: ${analytical_mean:.2f}")
    print(f"  Expected std: ${analytical_std:.2f}")
    
    # Confidence intervals
    ci_lower, ci_upper = gbm.get_confidence_intervals(initial_value, days/365.25)
    print(f"  95% Confidence interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
    
    print("\nThis framework is ready for delta hedging Monte Carlo analysis!")
    print("Next steps: Implement delta hedging logic at each time step.")
    
else:
    print("Failed to fetch sufficient data for demonstration")
