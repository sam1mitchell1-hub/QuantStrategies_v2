"""
Geometric Brownian Motion (GBM) implementation.

This module implements the Geometric Brownian Motion stochastic process,
commonly used in the Black-Scholes model for options pricing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy import stats
from .base import StochasticProcess


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion stochastic process.
    
    The GBM process follows the SDE:
    dS = μS dt + σS dW
    
    Where:
    - S is the asset price
    - μ is the drift (expected return)
    - σ is the volatility
    - dW is a Wiener process (Brownian motion)
    """
    
    def __init__(self, mu: Optional[float] = None, sigma: Optional[float] = None):
        """
        Initialize the GBM process.
        
        Args:
            mu: Drift parameter (if None, will be fitted from data)
            sigma: Volatility parameter (if None, will be fitted from data)
        """
        super().__init__("Geometric Brownian Motion")
        self.mu = mu
        self.sigma = sigma
        self.parameters = {"mu": mu, "sigma": sigma}
        
    def get_required_parameters(self) -> list:
        """Get required parameters for GBM."""
        return ["mu", "sigma"]
    
    def fit(self, data: pd.Series, method: str = "mle") -> Dict[str, Any]:
        """
        Fit GBM parameters to historical data.
        
        Args:
            data: Historical price data (pandas Series with datetime index)
            method: Fitting method ("mle" for maximum likelihood, "moments" for method of moments)
            
        Returns:
            Dictionary containing fitted parameters and fit statistics
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to fit GBM")
        
        # Calculate log returns
        log_prices = np.log(data)
        log_returns = log_prices.diff().dropna()
        
        # Calculate time step (assume daily data if no frequency is set)
        if data.index.freq is not None:
            dt = data.index.freq.delta.total_seconds() / (365.25 * 24 * 3600)  # Convert to years
        else:
            # Assume daily data if no frequency is set
            dt = 1.0 / 365.25
        
        if method == "mle":
            # Maximum likelihood estimation
            mu_hat = log_returns.mean() / dt  # Annualized
            sigma_hat = log_returns.std() / np.sqrt(dt)  # Annualized
        elif method == "moments":
            # Method of moments
            mu_hat = log_returns.mean() / dt
            sigma_hat = log_returns.std() / np.sqrt(dt)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        # Store fitted parameters (ensure they are scalars)
        self.mu = float(mu_hat)
        self.sigma = float(sigma_hat)
        self.parameters = {"mu": self.mu, "sigma": self.sigma}
        self.fitted = True
        self.fitted_data = data
        
        # Calculate fit statistics (use the same dt as above)
        theoretical_returns = (self.mu - 0.5 * self.sigma**2) * dt
        theoretical_variance = self.sigma**2 * dt
        
        actual_returns = log_returns.mean()
        actual_variance = log_returns.var()
        
        # Log-likelihood
        log_likelihood = -0.5 * len(log_returns) * np.log(2 * np.pi * theoretical_variance) - \
                        0.5 * np.sum((log_returns - theoretical_returns)**2) / theoretical_variance
        
        return {
            "mu": mu_hat,
            "sigma": sigma_hat,
            "log_likelihood": log_likelihood,
            "theoretical_mean": theoretical_returns,
            "actual_mean": actual_returns,
            "theoretical_variance": theoretical_variance,
            "actual_variance": actual_variance,
            "method": method,
            "data_points": len(data),
            "log_returns_points": len(log_returns)
        }
    
    def step(self, current_value: float, dt: float, random_shock: Optional[float] = None) -> float:
        """
        Step the GBM process forward by one time increment.
        
        Args:
            current_value: Current value of the process
            dt: Time step size (in years)
            random_shock: Optional random shock (if None, will be generated from N(0,1))
            
        Returns:
            New value of the process after one step
        """
        if not self.fitted and (self.mu is None or self.sigma is None):
            raise ValueError("Process must be fitted or parameters must be set")
        
        if random_shock is None:
            random_shock = np.random.standard_normal()
        
        # GBM formula: S(t+dt) = S(t) * exp((μ - 0.5*σ²)dt + σ*sqrt(dt)*Z)
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * random_shock
        
        return current_value * np.exp(drift + diffusion)
    
    def simulate_path(self, 
                     initial_value: float, 
                     time_steps: np.ndarray, 
                     random_shocks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate a complete path of the GBM process.
        
        Args:
            initial_value: Starting value of the process
            time_steps: Array of time points for the simulation
            random_shocks: Optional array of pre-generated random shocks
            
        Returns:
            Array of simulated values
        """
        if not self.fitted and (self.mu is None or self.sigma is None):
            raise ValueError("Process must be fitted or parameters must be set")
        
        n_steps = len(time_steps)
        path = np.zeros(n_steps)
        path[0] = initial_value
        
        if random_shocks is None:
            random_shocks = np.random.standard_normal(n_steps - 1)
        
        for i in range(1, n_steps):
            dt = time_steps[i] - time_steps[i-1]
            path[i] = self.step(path[i-1], dt, random_shocks[i-1])
        
        return path
    
    def get_analytical_mean(self, initial_value: float, time: float) -> float:
        """
        Get the analytical expected value at time t.
        
        Args:
            initial_value: Initial value of the process
            time: Time horizon (in years)
            
        Returns:
            Expected value at time t
        """
        return initial_value * np.exp(self.mu * time)
    
    def get_analytical_variance(self, initial_value: float, time: float) -> float:
        """
        Get the analytical variance at time t.
        
        Args:
            initial_value: Initial value of the process
            time: Time horizon (in years)
            
        Returns:
            Variance at time t
        """
        return (initial_value**2) * np.exp(2 * self.mu * time) * (np.exp(self.sigma**2 * time) - 1)
    
    def get_confidence_intervals(self, initial_value: float, time: float, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence intervals for the process at time t.
        
        Args:
            initial_value: Initial value of the process
            time: Time horizon (in years)
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        mean = self.get_analytical_mean(initial_value, time)
        std = np.sqrt(self.get_analytical_variance(initial_value, time))
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return lower, upper
