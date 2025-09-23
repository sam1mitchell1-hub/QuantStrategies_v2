"""
PDE Solvers Module

This module provides finite difference solvers for partial differential equations
used in quantitative finance, particularly for options pricing.

Main Components:
- Base classes for PDE solvers
- Black-Scholes equation solvers
- Utility functions for grid generation and matrix operations
"""

from .solvers.black_scholes_cn import BlackScholesCNSolver

__all__ = ['BlackScholesCNSolver']
