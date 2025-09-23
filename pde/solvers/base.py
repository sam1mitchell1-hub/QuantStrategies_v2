"""
Base classes for PDE solvers.

This module defines abstract base classes for implementing finite difference
solvers for partial differential equations used in quantitative finance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class PDESolver(ABC):
    """
    Abstract base class for all PDE solvers.
    
    This class defines the common interface that all PDE solvers must implement,
    providing a consistent API for different solution methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the PDE solver.
        
        Args:
            name: Human-readable name of the solver
        """
        self.name = name
        self.solved = False
        self.solution = None
        self.grid_info = {}
        
    @abstractmethod
    def setup_grid(self) -> None:
        """Set up the computational grid."""
        pass
    
    @abstractmethod
    def apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to the grid."""
        pass
    
    @abstractmethod
    def apply_initial_conditions(self) -> None:
        """Apply initial/terminal conditions to the grid."""
        pass
    
    @abstractmethod
    def solve(self) -> np.ndarray:
        """
        Solve the PDE.
        
        Returns:
            Solution array
        """
        pass
    
    @abstractmethod
    def get_solution_at_point(self, *args) -> float:
        """
        Get solution value at a specific point.
        
        Args:
            *args: Coordinates of the point
            
        Returns:
            Solution value at the point
        """
        pass
    
    def get_solution_info(self) -> Dict[str, Any]:
        """
        Get information about the solved PDE.
        
        Returns:
            Dictionary containing solution metadata
        """
        return {
            'solver_name': self.name,
            'solved': self.solved,
            'grid_info': self.grid_info,
            'solution_shape': self.solution.shape if self.solution is not None else None
        }


class BlackScholesSolver(PDESolver):
    """
    Abstract base class for Black-Scholes equation solvers.
    
    This class provides common functionality for all Black-Scholes PDE solvers,
    including parameter validation and common boundary conditions.
    """
    
    def __init__(self, 
                 S_min: float,
                 S_max: float,
                 T: float,
                 r: float,
                 sigma: float,
                 K: float,
                 option_type: str,
                 N_S: int,
                 N_T: int,
                 name: str = "Black-Scholes Solver"):
        """
        Initialize the Black-Scholes solver.
        
        Args:
            S_min: Minimum underlying price (typically 0)
            S_max: Maximum underlying price
            T: Time to expiration
            r: Risk-free interest rate
            sigma: Volatility
            K: Strike price
            option_type: 'call' or 'put'
            N_S: Number of spatial grid points
            N_T: Number of time steps
            name: Solver name
        """
        super().__init__(name)
        
        # Validate parameters
        self._validate_parameters(S_min, S_max, T, r, sigma, K, option_type, N_S, N_T)
        
        # Store parameters
        self.S_min = S_min
        self.S_max = S_max
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K = K
        self.option_type = option_type.lower()
        self.N_S = N_S
        self.N_T = N_T
        
        # Initialize grids
        self.S_grid = None
        self.t_grid = None
        self.x_grid = None  # Log-space grid
        
    def _validate_parameters(self, S_min, S_max, T, r, sigma, K, option_type, N_S, N_T):
        """Validate input parameters."""
        if S_min < 0:
            raise ValueError("S_min must be non-negative")
        if S_max <= S_min:
            raise ValueError("S_max must be greater than S_min")
        if T <= 0:
            raise ValueError("T must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if K <= 0:
            raise ValueError("K must be positive")
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if N_S < 2:
            raise ValueError("N_S must be at least 2")
        if N_T < 1:
            raise ValueError("N_T must be at least 1")
    
    def get_payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff function.
        
        Args:
            S: Underlying price array
            
        Returns:
            Payoff array
        """
        if self.option_type == 'call':
            return np.maximum(S - self.K, 0)
        else:  # put
            return np.maximum(self.K - S, 0)
    
    def get_boundary_condition_at_zero(self, t: float) -> float:
        """
        Get boundary condition at S=0.
        
        Args:
            t: Time
            
        Returns:
            Boundary value at S=0
        """
        if self.option_type == 'call':
            return 0.0
        else:  # put
            return self.K * np.exp(-self.r * (self.T - t))
    
    def get_boundary_condition_at_infinity(self, t: float) -> float:
        """
        Get boundary condition at S→∞.
        
        Args:
            t: Time
            
        Returns:
            Boundary value at S→∞
        """
        if self.option_type == 'call':
            # For large S, call option value ≈ S - K*exp(-r*(T-t))
            return self.S_max - self.K * np.exp(-self.r * (self.T - t))
        else:  # put
            return 0.0
    
    def get_solution_at_point(self, S: float, t: float) -> float:
        """
        Get solution value at a specific (S, t) point.
        
        Args:
            S: Underlying price
            t: Time
            
        Returns:
            Option value at (S, t)
        """
        if not self.solved:
            raise ValueError("PDE must be solved before querying solution")
        
        # Find closest grid points and interpolate
        S_idx = np.searchsorted(self.S_grid, S)
        t_idx = np.searchsorted(self.t_grid, t)
        
        # Handle boundary cases
        S_idx = max(0, min(S_idx - 1, self.N_S - 2))
        t_idx = max(0, min(t_idx - 1, self.N_T - 2))
        
        # Simple linear interpolation
        S1, S2 = self.S_grid[S_idx], self.S_grid[S_idx + 1]
        t1, t2 = self.t_grid[t_idx], self.t_grid[t_idx + 1]
        
        # Bilinear interpolation
        w_S = (S - S1) / (S2 - S1) if S2 > S1 else 0
        w_t = (t - t1) / (t2 - t1) if t2 > t1 else 0
        
        V11 = self.solution[t_idx, S_idx]
        V12 = self.solution[t_idx, S_idx + 1]
        V21 = self.solution[t_idx + 1, S_idx]
        V22 = self.solution[t_idx + 1, S_idx + 1]
        
        V1 = V11 * (1 - w_S) + V12 * w_S
        V2 = V21 * (1 - w_S) + V22 * w_S
        
        return V1 * (1 - w_t) + V2 * w_t
