"""
Crank-Nicolson solver with Rannacher smoothing for the Black-Scholes PDE.

This solver uses the Crank-Nicolson scheme with Rannacher smoothing in the first
few time steps to improve accuracy near the payoff discontinuity.
"""

import numpy as np
from typing import Dict, Any
from .black_scholes_cn import BlackScholesCNSolver
from ..utils.grid_utils import create_log_grid, create_time_grid
from ..utils.matrix_utils import thomas_algorithm


class BlackScholesCNRannacherSolver(BlackScholesCNSolver):
    """
    Crank-Nicolson solver with Rannacher smoothing for the Black-Scholes PDE.
    
    The Rannacher smoothing technique uses implicit Euler for the first few time steps
    to smooth out the discontinuity at the payoff, then switches to Crank-Nicolson.
    This improves accuracy near the strike price.
    """
    
    def __init__(self, 
                 S_min: float = 0.0,
                 S_max: float = None,
                 T: float = 1.0,
                 r: float = 0.05,
                 sigma: float = 0.2,
                 K: float = 100.0,
                 option_type: str = 'call',
                 N_S: int = 100,
                 N_T: int = 100,
                 rannacher_steps: int = 4,
                 q: float = 0.0):
        """
        Initialize the Crank-Nicolson solver with Rannacher smoothing.
        
        Args:
            S_min: Minimum underlying price (default: 0.0)
            S_max: Maximum underlying price (default: 4 * K)
            T: Time to expiration (default: 1.0)
            r: Risk-free interest rate (default: 0.05)
            sigma: Volatility (default: 0.2)
            K: Strike price (default: 100.0)
            option_type: 'call' or 'put' (default: 'call')
            N_S: Number of spatial grid points (default: 100)
            N_T: Number of time steps (default: 100)
            rannacher_steps: Number of Rannacher smoothing steps (default: 4)
            q: Continuous dividend/borrow yield (default: 0.0)
        """
        # Set default S_max if not provided
        if S_max is None:
            S_max = 4 * K
            
        super().__init__(S_min, S_max, T, r, sigma, K, option_type, N_S, N_T, q)
        
        # Rannacher smoothing parameters
        self.rannacher_steps = rannacher_steps
        self.theta_rannacher = 1.0  # Implicit Euler for Rannacher steps
        self.theta_cn = 0.5  # Crank-Nicolson for remaining steps
        
        # Grid and solution arrays
        self.x_grid = None
        self.dx = None
        self.dt = None
        self.solution = None
        
    def setup_grid(self) -> None:
        """Set up the computational grid in log-space."""
        # Create log-space grid
        self.S_grid, self.x_grid = create_log_grid(self.S_min, self.S_max, self.N_S)
        
        # Create time grid
        self.t_grid = create_time_grid(self.T, self.N_T)
        
        # Calculate grid spacing
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dt = self.t_grid[1] - self.t_grid[0]
        
        # Initialize solution array (time x space)
        self.solution = np.zeros((self.N_T + 1, self.N_S + 1))
        
        # Store grid information
        self.grid_info = {
            'S_min': self.S_min,
            'S_max': self.S_max,
            'N_S': self.N_S,
            'N_T': self.N_T,
            'dx': self.dx,
            'dt': self.dt,
            'x_min': self.x_grid[0],
            'x_max': self.x_grid[-1],
            'rannacher_steps': self.rannacher_steps
        }
        
    def apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to the grid."""
        # Boundary conditions are applied during the time stepping
        # This method is kept for interface consistency
        pass
        
    def apply_initial_conditions(self) -> None:
        """Apply terminal conditions (payoff) to the grid."""
        # Set terminal condition (payoff at expiration)
        self.solution[-1, :] = self.get_payoff(self.S_grid)
        
    def _get_coefficients(self, theta: float) -> tuple:
        """
        Get the coefficients for the finite difference scheme.
        
        Args:
            theta: Scheme parameter (1.0 for implicit Euler, 0.5 for Crank-Nicolson)
            
        Returns:
            Tuple of (a, b, c) coefficients for the tridiagonal system
        """
        # Spatial derivatives coefficients
        alpha = 0.5 * self.sigma**2 / (self.dx**2)
        beta = (self.r - 0.5 * self.sigma**2) / (2 * self.dx)
        gamma = self.r
        
        # Finite difference coefficients
        # For the tridiagonal system: a[i]*V[i-1] + b[i]*V[i] + c[i]*V[i+1] = d[i]
        
        # Lower diagonal (a[i] for i = 1, ..., N_S-1)
        a = np.zeros(self.N_S)
        a[1:] = -theta * self.dt * (alpha - beta)  # Skip first element (boundary)
        
        # Main diagonal (b[i] for i = 0, ..., N_S)
        b = np.ones(self.N_S + 1)
        b[1:-1] = 1 + theta * self.dt * (2 * alpha + gamma)
        
        # Upper diagonal (c[i] for i = 0, ..., N_S-1)
        c = np.zeros(self.N_S)
        c[:-1] = -theta * self.dt * (alpha + beta)
        
        return a, b, c
        
    def _get_rhs_vector(self, V_old: np.ndarray, theta: float) -> np.ndarray:
        """
        Get the right-hand side vector for the finite difference scheme.
        
        Args:
            V_old: Solution at previous time step
            theta: Scheme parameter
            
        Returns:
            Right-hand side vector
        """
        # Spatial derivatives coefficients
        alpha = 0.5 * self.sigma**2 / (self.dx**2)
        beta = (self.r - 0.5 * self.sigma**2) / (2 * self.dx)
        gamma = self.r
        
        # Initialize RHS
        d = np.zeros(self.N_S + 1)
        
        # Interior points (i = 1, ..., N_S-1)
        for i in range(1, self.N_S):
            # Explicit part of the scheme
            explicit_term = (1 - theta) * self.dt * (
                alpha * (V_old[i-1] - 2*V_old[i] + V_old[i+1]) +
                beta * (V_old[i+1] - V_old[i-1]) -
                gamma * V_old[i]
            )
            d[i] = V_old[i] + explicit_term
            
        return d
        
    def _apply_boundary_conditions_to_rhs(self, d: np.ndarray, t: float) -> np.ndarray:
        """
        Apply boundary conditions to the RHS vector.
        
        Args:
            d: RHS vector
            t: Current time
            
        Returns:
            Modified RHS vector with boundary conditions
        """
        # Boundary at S_min (x_min)
        d[0] = self.get_boundary_condition_at_zero(t)
        
        # Boundary at S_max (x_max)
        d[-1] = self.get_boundary_condition_at_infinity(t)
        
        return d
        
    def solve(self) -> np.ndarray:
        """
        Solve the Black-Scholes PDE using Crank-Nicolson with Rannacher smoothing.
        
        Returns:
            Solution array (time x space)
        """
        if self.solution is None:
            raise ValueError("Grid must be set up before solving")
            
        # Apply terminal condition
        self.apply_initial_conditions()
        
        # Time stepping (backwards from expiration to present)
        for n in range(self.N_T - 1, -1, -1):
            t = self.t_grid[n]
            
            # Determine which scheme to use
            if n >= self.N_T - self.rannacher_steps:
                # Use Rannacher smoothing (implicit Euler) for first few steps
                theta = self.theta_rannacher
                scheme_name = "Rannacher (Implicit Euler)"
            else:
                # Use Crank-Nicolson for remaining steps
                theta = self.theta_cn
                scheme_name = "Crank-Nicolson"
            
            # Get coefficients for the chosen scheme
            a, b, c = self._get_coefficients(theta)
            
            # Get RHS vector
            d = self._get_rhs_vector(self.solution[n + 1], theta)
            
            # Apply boundary conditions
            d = self._apply_boundary_conditions_to_rhs(d, t)
            
            # Solve tridiagonal system
            self.solution[n, :] = thomas_algorithm(a, b, c, d)
            
        self.solved = True
        return self.solution
        
    def get_option_price(self, S: float, t: float = 0.0) -> float:
        """
        Get option price at a specific underlying price and time.
        
        Args:
            S: Underlying price
            t: Time (default: 0.0 for current time)
            
        Returns:
            Option price
        """
        return self.get_solution_at_point(S, t)
        
    def get_greeks(self, S: float, t: float = 0.0) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            S: Underlying price
            t: Time (default: 0.0 for current time)
            
        Returns:
            Dictionary containing Delta, Gamma, Theta
        """
        if not self.solved:
            raise ValueError("PDE must be solved before calculating Greeks")
            
        # Small perturbation for finite differences
        dS = S * 0.01  # 1% of underlying price
        dt = 0.01  # Small time step
        
        # Delta: ∂V/∂S
        V_plus = self.get_option_price(S + dS, t)
        V_minus = self.get_option_price(S - dS, t)
        delta = (V_plus - V_minus) / (2 * dS)
        
        # Gamma: ∂²V/∂S²
        V_center = self.get_option_price(S, t)
        gamma = (V_plus - 2*V_center + V_minus) / (dS**2)
        
        # Theta: ∂V/∂t
        V_future = self.get_option_price(S, t + dt)
        theta = (V_future - V_center) / dt
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta
        }
        
    def get_solution_surface(self) -> Dict[str, np.ndarray]:
        """
        Get the complete solution surface.
        
        Returns:
            Dictionary containing S_grid, t_grid, and solution array
        """
        if not self.solved:
            raise ValueError("PDE must be solved before getting solution surface")
            
        return {
            'S_grid': self.S_grid,
            't_grid': self.t_grid,
            'solution': self.solution
        }
        
    def get_rannacher_info(self) -> Dict[str, Any]:
        """
        Get information about the Rannacher smoothing.
        
        Returns:
            Dictionary containing Rannacher smoothing information
        """
        return {
            'rannacher_steps': self.rannacher_steps,
            'theta_rannacher': self.theta_rannacher,
            'theta_cn': self.theta_cn,
            'rannacher_time_steps': list(range(self.N_T - self.rannacher_steps, self.N_T)),
            'cn_time_steps': list(range(0, self.N_T - self.rannacher_steps))
        }
