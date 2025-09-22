"""
Unit tests for the stochastic processes framework.

This module contains tests that can run in CI without matplotlib dependencies.
"""

import pytest
import numpy as np
import pandas as pd
from stochastic import GeometricBrownianMotion


class TestGeometricBrownianMotion:
    """Test cases for Geometric Brownian Motion implementation."""
    
    def test_initialization(self):
        """Test GBM initialization with and without parameters."""
        # Test with parameters
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        assert gbm.mu == 0.1
        assert gbm.sigma == 0.2
        assert gbm.parameters["mu"] == 0.1
        assert gbm.parameters["sigma"] == 0.2
        
        # Test without parameters
        gbm_empty = GeometricBrownianMotion()
        assert gbm_empty.mu is None
        assert gbm_empty.sigma is None
        assert not gbm_empty.fitted
    
    def test_get_required_parameters(self):
        """Test getting required parameters."""
        gbm = GeometricBrownianMotion()
        params = gbm.get_required_parameters()
        assert params == ["mu", "sigma"]
    
    def test_fit_synthetic_data(self):
        """Test fitting GBM to synthetic data."""
        # Generate synthetic data
        np.random.seed(42)  # For reproducible tests
        true_mu = 0.08
        true_sigma = 0.25
        initial_price = 100.0
        days = 1000
        
        # Generate synthetic GBM data
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        dt = 1/365.25
        random_shocks = np.random.standard_normal(days-1)
        
        prices = np.zeros(days)
        prices[0] = initial_price
        
        for i in range(1, days):
            drift = (true_mu - 0.5 * true_sigma**2) * dt
            diffusion = true_sigma * np.sqrt(dt) * random_shocks[i-1]
            prices[i] = prices[i-1] * np.exp(drift + diffusion)
        
        data = pd.Series(prices, index=dates, name='close')
        
        # Fit GBM
        gbm = GeometricBrownianMotion()
        fit_results = gbm.fit(data, method="mle")
        
        # Check that parameters are reasonable
        assert abs(fit_results['mu'] - true_mu) < 0.1  # Within 10% of true value
        assert abs(fit_results['sigma'] - true_sigma) < 0.1
        assert gbm.fitted
        assert fit_results['data_points'] == days
        assert fit_results['log_returns_points'] == days - 1
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        gbm = GeometricBrownianMotion()
        data = pd.Series([100.0], index=[pd.Timestamp('2020-01-01')])
        
        with pytest.raises(ValueError, match="Need at least 2 data points"):
            gbm.fit(data)
    
    def test_fit_invalid_method(self):
        """Test fitting with invalid method."""
        gbm = GeometricBrownianMotion()
        data = pd.Series([100.0, 101.0], index=pd.date_range('2020-01-01', periods=2))
        
        with pytest.raises(ValueError, match="Unknown fitting method"):
            gbm.fit(data, method="invalid")
    
    def test_step_with_parameters(self):
        """Test stepping the process with pre-set parameters."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        current_value = 100.0
        dt = 1/365.25  # Daily step
        
        # Test with provided random shock
        random_shock = 0.5
        new_value = gbm.step(current_value, dt, random_shock)
        
        # Verify the step calculation
        expected_drift = (0.1 - 0.5 * 0.2**2) * dt
        expected_diffusion = 0.2 * np.sqrt(dt) * random_shock
        expected_value = current_value * np.exp(expected_drift + expected_diffusion)
        
        assert abs(new_value - expected_value) < 1e-10
    
    def test_step_without_parameters(self):
        """Test stepping without fitted parameters."""
        gbm = GeometricBrownianMotion()
        
        with pytest.raises(ValueError, match="Process must be fitted or parameters must be set"):
            gbm.step(100.0, 0.01)
    
    def test_simulate_path(self):
        """Test simulating a complete path."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        initial_value = 100.0
        time_steps = np.linspace(0, 1, 253)  # One year, daily steps
        
        path = gbm.simulate_path(initial_value, time_steps)
        
        assert len(path) == len(time_steps)
        assert path[0] == initial_value
        assert all(np.isfinite(path))  # All values should be finite
    
    def test_simulate_path_with_shocks(self):
        """Test simulating path with pre-generated random shocks."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        initial_value = 100.0
        time_steps = np.linspace(0, 1, 11)  # 10 steps
        random_shocks = np.array([0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1, -0.2, 0.0])
        
        path = gbm.simulate_path(initial_value, time_steps, random_shocks)
        
        assert len(path) == len(time_steps)
        assert path[0] == initial_value
        assert all(np.isfinite(path))
    
    def test_analytical_mean(self):
        """Test analytical mean calculation."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        initial_value = 100.0
        time = 1.0  # One year
        
        mean = gbm.get_analytical_mean(initial_value, time)
        expected = initial_value * np.exp(0.1 * time)
        
        assert abs(mean - expected) < 1e-10
    
    def test_analytical_variance(self):
        """Test analytical variance calculation."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        initial_value = 100.0
        time = 1.0  # One year
        
        variance = gbm.get_analytical_variance(initial_value, time)
        expected = (initial_value**2) * np.exp(2 * 0.1 * time) * (np.exp(0.2**2 * time) - 1)
        
        assert abs(variance - expected) < 1e-10
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2)
        initial_value = 100.0
        time = 1.0
        confidence = 0.95
        
        lower, upper = gbm.get_confidence_intervals(initial_value, time, confidence)
        
        assert lower < upper
        assert lower > 0  # Price should be positive
        assert upper > 0
        
        # Test with different confidence levels
        lower_99, upper_99 = gbm.get_confidence_intervals(initial_value, time, 0.99)
        assert lower_99 < lower  # 99% CI should be wider
        assert upper_99 > upper
    
    def test_fit_methods_consistency(self):
        """Test that MLE and moments methods give similar results."""
        # Generate synthetic data
        np.random.seed(42)
        true_mu = 0.08
        true_sigma = 0.25
        initial_price = 100.0
        days = 1000
        
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        dt = 1/365.25
        random_shocks = np.random.standard_normal(days-1)
        
        prices = np.zeros(days)
        prices[0] = initial_price
        
        for i in range(1, days):
            drift = (true_mu - 0.5 * true_sigma**2) * dt
            diffusion = true_sigma * np.sqrt(dt) * random_shocks[i-1]
            prices[i] = prices[i-1] * np.exp(drift + diffusion)
        
        data = pd.Series(prices, index=dates, name='close')
        
        # Fit with both methods
        gbm_mle = GeometricBrownianMotion()
        gbm_moments = GeometricBrownianMotion()
        
        results_mle = gbm_mle.fit(data, method="mle")
        results_moments = gbm_moments.fit(data, method="moments")
        
        # Results should be very similar for large datasets
        assert abs(results_mle['mu'] - results_moments['mu']) < 0.01
        assert abs(results_mle['sigma'] - results_moments['sigma']) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
