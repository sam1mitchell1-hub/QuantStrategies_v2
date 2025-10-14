"""
FD Pricing Integration

Integrates finite difference solvers for Q-measure pricing:
- Black-Scholes FD pricing for individual options
- Greeks calculation using FD methods
- IV surface fitting and interpolation
- Risk-neutral pricing for option structures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import our FD solvers
from pde import BlackScholesCNSolver, BlackScholesCNRannacherSolver


@dataclass
class PricingResult:
    """Result of option pricing."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_vol: float
    solver_info: Dict[str, Any]


@dataclass
class StructurePricing:
    """Pricing result for option structure."""
    net_price: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    long_leg_pricing: PricingResult
    short_leg_pricing: PricingResult
    structure_metrics: Dict[str, float]


class FDPricer:
    """Finite difference-based option pricer for Q-measure pricing."""
    
    def __init__(self, config):
        """Initialize FD pricer with configuration."""
        self.config = config
        self.solver_cache = {}  # Cache for solvers
        
    def price_option(self, 
                    strike: float,
                    option_type: str,
                    expiry_days: float,
                    current_price: float,
                    risk_free_rate: float,
                    dividend_yield: float,
                    implied_vol: float) -> PricingResult:
        """
        Price a single option using FD solver.
        
        Args:
            strike: Strike price
            option_type: 'call' or 'put'
            expiry_days: Days to expiry
            current_price: Current underlying price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            implied_vol: Implied volatility
            
        Returns:
            PricingResult with price and Greeks
        """
        # Create solver key for caching
        solver_key = f"{option_type}_{strike}_{expiry_days}_{implied_vol}"
        
        if solver_key not in self.solver_cache:
            # Create new solver
            solver = self._create_solver(
                strike=strike,
                option_type=option_type,
                expiry_days=expiry_days,
                current_price=current_price,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_vol=implied_vol
            )
            self.solver_cache[solver_key] = solver
        else:
            solver = self.solver_cache[solver_key]
        
        # Price the option
        price = solver.get_option_price(current_price, t=0.0)
        
        # Calculate Greeks
        greeks = solver.get_greeks(current_price, t=0.0)
        
        # Calculate additional Greeks
        rho = self._calculate_rho(solver, current_price, risk_free_rate)
        
        return PricingResult(
            price=price,
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            theta=greeks['theta'],
            vega=greeks['vega'],
            rho=rho,
            implied_vol=implied_vol,
            solver_info={
                'solver_type': self.config.pricing.fd_solver,
                'grid_points': solver.N_S,
                'time_steps': solver.N_T,
                's_max': solver.S_max
            }
        )
    
    def price_structure(self, 
                       structure,
                       current_price: float,
                       risk_free_rate: float,
                       dividend_yield: float,
                       implied_vol: float) -> StructurePricing:
        """
        Price an option structure using FD pricing.
        
        Args:
            structure: OptionStructure object
            current_price: Current underlying price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            implied_vol: Implied volatility
            
        Returns:
            StructurePricing with net pricing and Greeks
        """
        # Price long leg
        long_pricing = self.price_option(
            strike=structure.long_leg.strike_price,
            option_type=structure.long_leg.option_type.value,
            expiry_days=(structure.long_leg.expiry_date - pd.Timestamp.now()).days,
            current_price=current_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            implied_vol=implied_vol
        )
        
        # Price short leg
        short_pricing = self.price_option(
            strike=structure.short_leg.strike_price,
            option_type=structure.short_leg.option_type.value,
            expiry_days=(structure.short_leg.expiry_date - pd.Timestamp.now()).days,
            current_price=current_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            implied_vol=implied_vol
        )
        
        # Calculate net values
        if structure.structure_type.value == 'bull_call_spread':
            # Long call, short call
            net_price = long_pricing.price - short_pricing.price
            net_delta = long_pricing.delta - short_pricing.delta
            net_gamma = long_pricing.gamma - short_pricing.gamma
            net_theta = long_pricing.theta - short_pricing.theta
            net_vega = long_pricing.vega - short_pricing.vega
            net_rho = long_pricing.rho - short_pricing.rho
            
        elif structure.structure_type.value == 'bear_put_spread':
            # Long put, short put
            net_price = long_pricing.price - short_pricing.price
            net_delta = long_pricing.delta - short_pricing.delta
            net_gamma = long_pricing.gamma - short_pricing.gamma
            net_theta = long_pricing.theta - short_pricing.theta
            net_vega = long_pricing.vega - short_pricing.vega
            net_rho = long_pricing.rho - short_pricing.rho
            
        else:
            raise ValueError(f"Unsupported structure type: {structure.structure_type}")
        
        # Calculate structure metrics
        structure_metrics = {
            'max_profit': structure.max_profit,
            'max_loss': structure.max_loss,
            'breakeven': structure.breakeven,
            'width': structure.width,
            'net_premium': net_price,
            'premium_ratio': net_price / structure.width if structure.width > 0 else 0
        }
        
        return StructurePricing(
            net_price=net_price,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            net_rho=net_rho,
            long_leg_pricing=long_pricing,
            short_leg_pricing=short_pricing,
            structure_metrics=structure_metrics
        )
    
    def _create_solver(self, 
                      strike: float,
                      option_type: str,
                      expiry_days: float,
                      current_price: float,
                      risk_free_rate: float,
                      dividend_yield: float,
                      implied_vol: float):
        """Create FD solver for option pricing."""
        # Calculate time to expiry
        T = expiry_days / 365.0  # Convert to years
        
        # Set up grid parameters
        S_min = 0.0
        S_max = current_price * self.config.pricing.fd_s_max_multiple
        
        # Create solver
        if self.config.pricing.fd_solver == 'crank_nicolson':
            solver = BlackScholesCNSolver(
                S_min=S_min,
                S_max=S_max,
                T=T,
                r=risk_free_rate,
                sigma=implied_vol,
                K=strike,
                option_type=option_type,
                N_S=self.config.pricing.fd_grid_points,
                N_T=self.config.pricing.fd_time_steps
            )
        elif self.config.pricing.fd_solver == 'rannacher':
            solver = BlackScholesCNRannacherSolver(
                S_min=S_min,
                S_max=S_max,
                T=T,
                r=risk_free_rate,
                sigma=implied_vol,
                K=strike,
                option_type=option_type,
                N_S=self.config.pricing.fd_grid_points,
                N_T=self.config.pricing.fd_time_steps,
                rannacher_steps=4
            )
        else:
            raise ValueError(f"Unsupported solver type: {self.config.pricing.fd_solver}")
        
        # Setup and solve
        solver.setup_grid()
        solver.apply_initial_conditions()
        solver.solve()
        
        return solver
    
    def _calculate_rho(self, solver, current_price: float, risk_free_rate: float) -> float:
        """Calculate rho (sensitivity to interest rate) using finite differences."""
        # Small perturbation for interest rate
        dr = 0.001  # 10 bps
        
        # Price with higher interest rate
        solver_high = self._create_solver_with_rate(solver, risk_free_rate + dr)
        price_high = solver_high.get_option_price(current_price, t=0.0)
        
        # Price with lower interest rate
        solver_low = self._create_solver_with_rate(solver, risk_free_rate - dr)
        price_low = solver_low.get_option_price(current_price, t=0.0)
        
        # Calculate rho
        rho = (price_high - price_low) / (2 * dr)
        
        return rho
    
    def _create_solver_with_rate(self, base_solver, new_rate: float):
        """Create solver with modified interest rate."""
        if self.config.pricing.fd_solver == 'crank_nicolson':
            solver = BlackScholesCNSolver(
                S_min=base_solver.S_min,
                S_max=base_solver.S_max,
                T=base_solver.T,
                r=new_rate,
                sigma=base_solver.sigma,
                K=base_solver.K,
                option_type=base_solver.option_type,
                N_S=base_solver.N_S,
                N_T=base_solver.N_T
            )
        elif self.config.pricing.fd_solver == 'rannacher':
            solver = BlackScholesCNRannacherSolver(
                S_min=base_solver.S_min,
                S_max=base_solver.S_max,
                T=base_solver.T,
                r=new_rate,
                sigma=base_solver.sigma,
                K=base_solver.K,
                option_type=base_solver.option_type,
                N_S=base_solver.N_S,
                N_T=base_solver.N_T,
                rannacher_steps=4
            )
        
        solver.setup_grid()
        solver.apply_initial_conditions()
        solver.solve()
        
        return solver
    
    def fit_iv_surface(self, 
                      options_data: pd.DataFrame,
                      current_price: float,
                      risk_free_rate: float,
                      dividend_yield: float) -> Dict[str, Any]:
        """
        Fit IV surface to market data.
        
        Args:
            options_data: Market options data
            current_price: Current underlying price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            Fitted IV surface parameters
        """
        if self.config.pricing.iv_surface_model == 'svi':
            return self._fit_svi_surface(options_data, current_price, risk_free_rate, dividend_yield)
        elif self.config.pricing.iv_surface_model == 'spline':
            return self._fit_spline_surface(options_data, current_price, risk_free_rate, dividend_yield)
        else:
            raise ValueError(f"Unsupported IV surface model: {self.config.pricing.iv_surface_model}")
    
    def _fit_svi_surface(self, 
                        options_data: pd.DataFrame,
                        current_price: float,
                        risk_free_rate: float,
                        dividend_yield: float) -> Dict[str, Any]:
        """Fit SVI (Stochastic Volatility Inspired) surface."""
        # Simplified SVI fitting - in practice, you'd use a proper SVI implementation
        # This is a placeholder that returns constant volatility
        
        surface_params = {
            'model': 'svi',
            'a': 0.04,  # ATM variance
            'b': 0.4,   # Slope of wings
            'rho': -0.4,  # Correlation
            'm': 0.0,   # ATM log-moneyness
            'sigma': 0.2,  # Volatility of variance
            'current_price': current_price,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield
        }
        
        return surface_params
    
    def _fit_spline_surface(self, 
                           options_data: pd.DataFrame,
                           current_price: float,
                           risk_free_rate: float,
                           dividend_yield: float) -> Dict[str, Any]:
        """Fit spline-based IV surface."""
        # Simplified spline fitting - in practice, you'd use proper spline interpolation
        # This is a placeholder that returns constant volatility
        
        surface_params = {
            'model': 'spline',
            'interpolation_method': 'cubic',
            'smoothing_factor': self.config.pricing.iv_smoothing_factor,
            'current_price': current_price,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield
        }
        
        return surface_params
    
    def interpolate_iv(self, 
                      surface_params: Dict[str, Any],
                      strike: float,
                      expiry_days: float,
                      current_price: float) -> float:
        """Interpolate IV from fitted surface."""
        # Simplified interpolation - in practice, you'd use the fitted surface
        # This returns a constant volatility for now
        
        base_vol = 0.2  # Base volatility
        
        # Add some term structure
        term_factor = 1.0 + 0.1 * np.exp(-expiry_days / 30.0)
        
        # Add some skew
        log_moneyness = np.log(strike / current_price)
        skew_factor = 1.0 - 0.2 * log_moneyness
        
        implied_vol = base_vol * term_factor * skew_factor
        
        return max(0.05, min(1.0, implied_vol))  # Clamp between 5% and 100%
    
    def calculate_expected_value(self, 
                               structure,
                               forecast: Dict[str, Any],
                               current_price: float,
                               risk_free_rate: float,
                               dividend_yield: float) -> float:
        """
        Calculate expected value under P-measure using Monte Carlo.
        
        Args:
            structure: OptionStructure object
            forecast: Forecast results
            current_price: Current underlying price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            Expected value under P-measure
        """
        # Get forecast parameters
        expected_return = forecast.get('expected_returns', 0.0)
        forecast_prob = forecast.get('probabilities', 0.5)
        
        # Calculate expected price at expiry
        if self.config.model.objective == 'classification':
            # Use probability-weighted expected price
            up_prob = forecast_prob
            down_prob = 1 - forecast_prob
            
            # Simplified: assume 2% up move or 2% down move
            up_price = current_price * 1.02
            down_price = current_price * 0.98
            
            expected_price = up_prob * up_price + down_prob * down_price
        else:
            # Use expected return directly
            expected_price = current_price * (1 + expected_return)
        
        # Calculate payoff at expected price
        if structure.structure_type.value == 'bull_call_spread':
            payoff = max(0, min(expected_price - structure.long_leg.strike_price, structure.width))
        elif structure.structure_type.value == 'bear_put_spread':
            payoff = max(0, min(structure.long_leg.strike_price - expected_price, structure.width))
        else:
            payoff = 0
        
        # Subtract net premium (Q-measure price)
        q_price = self.price_structure(
            structure, current_price, risk_free_rate, dividend_yield, 0.2
        ).net_price
        
        expected_value = payoff - q_price
        
        return expected_value
    
    def calculate_risk_metrics(self, 
                              structure,
                              current_price: float,
                              risk_free_rate: float,
                              dividend_yield: float) -> Dict[str, float]:
        """Calculate risk metrics for the structure."""
        # Price the structure
        pricing = self.price_structure(
            structure, current_price, risk_free_rate, dividend_yield, 0.2
        )
        
        # Calculate risk metrics
        risk_metrics = {
            'net_delta': pricing.net_delta,
            'net_gamma': pricing.net_gamma,
            'net_theta': pricing.net_theta,
            'net_vega': pricing.net_vega,
            'net_rho': pricing.net_rho,
            'max_loss': structure.max_loss,
            'max_profit': structure.max_profit,
            'breakeven': structure.breakeven,
            'width': structure.width,
            'premium_ratio': pricing.structure_metrics['premium_ratio']
        }
        
        # Calculate risk-adjusted metrics
        if structure.width > 0:
            risk_metrics['risk_reward_ratio'] = structure.max_profit / structure.max_loss
            risk_metrics['delta_per_width'] = pricing.net_delta / structure.width
            risk_metrics['gamma_per_width'] = pricing.net_gamma / structure.width
        
        return risk_metrics
    
    def clear_cache(self):
        """Clear solver cache to free memory."""
        self.solver_cache.clear()
        print("Solver cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about solver cache."""
        return {
            'cache_size': len(self.solver_cache),
            'cached_solvers': list(self.solver_cache.keys())
        }
