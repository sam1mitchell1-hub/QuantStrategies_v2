"""
Base class for stochastic processes.

This module defines the abstract base class that all stochastic processes
must implement, providing a common interface for fitting to historical data
and simulating future paths.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class StochasticProcess(ABC):
    """
    Abstract base class for stochastic processes.
    
    This class defines the interface that all stochastic processes must implement
    for fitting to historical data and simulating future paths. This is designed
    to work with the delta hedging Monte Carlo framework.
    """
    
    def __init__(self, name: str = "StochasticProcess"):
        """
        Initialize the stochastic process.
        
        Args:
            name: Human-readable name for the process
        """
        self.name = name
        self.fitted = False
        self.parameters = {}
        self.fitted_data = None
        
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Fit the stochastic process to historical data.
        
        Args:
            data: Historical price data (pandas Series with datetime index)
            **kwargs: Additional fitting parameters
            
        Returns:
            Dictionary containing fitted parameters and fit statistics
        """
        pass
    
    @abstractmethod
    def step(self, current_value: float, dt: float, random_shock: Optional[float] = None) -> float:
        """
        Step the process forward by one time increment.
        
        Args:
            current_value: Current value of the process
            dt: Time step size
            random_shock: Optional random shock (if None, will be generated)
            
        Returns:
            New value of the process after one step
        """
        pass
    
    @abstractmethod
    def simulate_path(self, 
                     initial_value: float, 
                     time_steps: np.ndarray, 
                     random_shocks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate a complete path of the process.
        
        Args:
            initial_value: Starting value of the process
            time_steps: Array of time points for the simulation
            random_shocks: Optional array of pre-generated random shocks
            
        Returns:
            Array of simulated values
        """
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> list:
        """
        Get list of required parameters for this process.
        
        Returns:
            List of parameter names required for this process
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate that all required parameters are present and valid.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        required = self.get_required_parameters()
        return all(param in parameters for param in required)
    
    def get_fit_summary(self) -> Dict[str, Any]:
        """
        Get summary of the fitting results.
        
        Returns:
            Dictionary containing fit summary information
        """
        if not self.fitted:
            return {"fitted": False, "message": "Process not yet fitted to data"}
        
        return {
            "fitted": True,
            "process_name": self.name,
            "parameters": self.parameters,
            "data_length": len(self.fitted_data) if self.fitted_data is not None else 0
        }
    
    def reset(self):
        """Reset the process to unfitted state."""
        self.fitted = False
        self.parameters = {}
        self.fitted_data = None
