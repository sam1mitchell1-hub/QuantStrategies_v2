"""
Grid generation utilities for PDE solvers.
"""

import numpy as np
from typing import Tuple


def create_log_grid(S_min: float, S_max: float, N_S: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a uniform grid in log-space.
    
    Args:
        S_min: Minimum underlying price
        S_max: Maximum underlying price
        N_S: Number of spatial grid points
        
    Returns:
        Tuple of (S_grid, x_grid) where x = ln(S)
    """
    # Handle S_min = 0 case by using a small positive value
    if S_min <= 0:
        S_min = 1e-10  # Very small positive value
    
    # Create log-space grid
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    x_grid = np.linspace(x_min, x_max, N_S)
    
    # Convert back to S-space
    S_grid = np.exp(x_grid)
    
    return S_grid, x_grid


def create_time_grid(T: float, N_T: int) -> np.ndarray:
    """
    Create a uniform time grid.
    
    Args:
        T: Time to expiration
        N_T: Number of time steps
        
    Returns:
        Time grid array
    """
    return np.linspace(0, T, N_T + 1)


def create_uniform_grid(min_val: float, max_val: float, N: int) -> np.ndarray:
    """
    Create a uniform grid between min_val and max_val.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        N: Number of grid points
        
    Returns:
        Uniform grid array
    """
    return np.linspace(min_val, max_val, N)
