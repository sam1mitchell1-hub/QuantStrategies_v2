"""
Trading Strategy Framework

A comprehensive options trading strategy framework that combines:
- Machine learning forecasting (P-measure)
- Finite difference option pricing (Q-measure)
- Risk management and position sizing
- Backtesting and production execution

Main Components:
- Configuration management
- Feature engineering pipeline
- GBT forecasting models
- Option structure selection
- FD-based pricing and Greeks
- Risk management
- Backtesting framework
"""

from .config import StrategyConfig
from .features import FeatureBuilder
from .forecasting import GBTForecaster
from .structures import OptionStructureSelector
from .pricing import FDPricer
from .risk import RiskManager
from .backtesting import Backtester

__all__ = [
    'StrategyConfig',
    'FeatureBuilder', 
    'GBTForecaster',
    'OptionStructureSelector',
    'FDPricer',
    'RiskManager',
    'Backtester'
]
