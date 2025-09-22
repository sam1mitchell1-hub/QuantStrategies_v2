"""
Stochastic process implementations.

This module contains concrete implementations of various stochastic processes
used in quantitative finance.
"""

from .base import StochasticProcess
from .gbm import GeometricBrownianMotion

__all__ = ['StochasticProcess', 'GeometricBrownianMotion']
