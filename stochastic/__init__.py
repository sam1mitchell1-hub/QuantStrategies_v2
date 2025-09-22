"""
Stochastic processes module for quantitative finance.

This module provides a framework for implementing and simulating various
stochastic processes used in quantitative finance, particularly for options
pricing and risk management.
"""

from .processes.base import StochasticProcess
from .processes.gbm import GeometricBrownianMotion

__all__ = ['StochasticProcess', 'GeometricBrownianMotion']
