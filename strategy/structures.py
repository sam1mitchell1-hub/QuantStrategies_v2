"""
Option Structure Selection

Selects appropriate option structures based on forecasts:
- Bull Call Spreads for bullish forecasts
- Bear Put Spreads for bearish forecasts
- Strike selection based on forecast bands
- Liquidity filtering and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class StructureType(Enum):
    """Structure type enumeration."""
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    NO_TRADE = "no_trade"


@dataclass
class Strike:
    """Option strike information."""
    strike_price: float
    option_type: OptionType
    expiry_date: pd.Timestamp
    bid: float
    ask: float
    mid: float
    open_interest: int
    volume: int
    implied_vol: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


@dataclass
class OptionStructure:
    """Option structure definition."""
    structure_type: StructureType
    long_leg: Strike
    short_leg: Strike
    net_premium: float
    max_profit: float
    max_loss: float
    breakeven: float
    width: float
    expected_value: Optional[float] = None
    sharpe_ratio: Optional[float] = None


@dataclass
class StructureSelection:
    """Structure selection result."""
    structure: Optional[OptionStructure]
    forecast_sharpe: float
    forecast_direction: str
    band_coefficient: float
    selection_reason: str
    liquidity_passed: bool


class OptionStructureSelector:
    """Selects option structures based on forecasts and market conditions."""
    
    def __init__(self, config):
        """Initialize structure selector with configuration."""
        self.config = config
        self.market_data = None
        self.iv_surface = None
        
    def select_structure(self, 
                        forecast: Dict[str, Any],
                        market_data: Dict[str, Any],
                        current_price: float,
                        current_time: pd.Timestamp) -> StructureSelection:
        """
        Select appropriate option structure based on forecast.
        
        Args:
            forecast: Forecast results from GBT model
            market_data: Current market data including options
            current_price: Current underlying price
            current_time: Current timestamp
            
        Returns:
            StructureSelection with selected structure
        """
        # Extract forecast information
        expected_return = forecast.get('expected_returns', 0.0)
        forecast_prob = forecast.get('probabilities', 0.5)
        
        # Calculate forecast Sharpe ratio
        forecast_sharpe = self._calculate_forecast_sharpe(expected_return, forecast_prob)
        
        # Check if we should trade
        if abs(forecast_sharpe) < self.config.structure.min_forecast_sharpe:
            return StructureSelection(
                structure=None,
                forecast_sharpe=forecast_sharpe,
                forecast_direction="neutral",
                band_coefficient=0.0,
                selection_reason="Forecast Sharpe below threshold",
                liquidity_passed=False
            )
        
        # Determine direction
        if forecast_sharpe > 0:
            direction = "bullish"
            structure_type = StructureType.BULL_CALL_SPREAD
        else:
            direction = "bearish"
            structure_type = StructureType.BEAR_PUT_SPREAD
        
        # Select strikes
        strikes = self._select_strikes(
            structure_type, 
            current_price, 
            expected_return,
            market_data,
            current_time
        )
        
        if strikes is None:
            return StructureSelection(
                structure=None,
                forecast_sharpe=forecast_sharpe,
                forecast_direction=direction,
                band_coefficient=0.0,
                selection_reason="No suitable strikes found",
                liquidity_passed=False
            )
        
        # Create structure
        structure = self._create_structure(structure_type, strikes, current_price)
        
        # Check liquidity requirements
        liquidity_passed = self._check_liquidity(structure, market_data)
        
        if not liquidity_passed:
            return StructureSelection(
                structure=None,
                forecast_sharpe=forecast_sharpe,
                forecast_direction=direction,
                band_coefficient=0.0,
                selection_reason="Liquidity requirements not met",
                liquidity_passed=False
            )
        
        return StructureSelection(
            structure=structure,
            forecast_sharpe=forecast_sharpe,
            forecast_direction=direction,
            band_coefficient=1.0,  # Default band coefficient
            selection_reason="Structure selected successfully",
            liquidity_passed=True
        )
    
    def _calculate_forecast_sharpe(self, expected_return: float, forecast_prob: float) -> float:
        """Calculate forecast Sharpe ratio."""
        # For classification models, convert probability to expected return
        if self.config.model.objective == 'classification':
            # Simple mapping: prob > 0.5 -> positive return, prob < 0.5 -> negative return
            # Scale by confidence (distance from 0.5)
            confidence = abs(forecast_prob - 0.5) * 2  # Scale to [0, 1]
            expected_return = (forecast_prob - 0.5) * confidence * 0.02  # Max 2% return
        
        # Calculate Sharpe ratio (simplified)
        # In practice, you'd use historical volatility or forecasted volatility
        volatility = 0.2  # Default 20% annualized volatility
        sharpe = expected_return / volatility if volatility > 0 else 0
        
        return sharpe
    
    def _select_strikes(self, 
                       structure_type: StructureType,
                       current_price: float,
                       expected_return: float,
                       market_data: Dict[str, Any],
                       current_time: pd.Timestamp) -> Optional[Tuple[Strike, Strike]]:
        """Select strikes for the given structure type."""
        # Get available options
        options = market_data.get('options', [])
        if not options:
            return None
        
        # Filter options by expiry and liquidity
        valid_options = self._filter_options(options, current_time)
        if len(valid_options) < 2:
            return None
        
        if structure_type == StructureType.BULL_CALL_SPREAD:
            return self._select_bull_call_strikes(current_price, expected_return, valid_options)
        elif structure_type == StructureType.BEAR_PUT_SPREAD:
            return self._select_bear_put_strikes(current_price, expected_return, valid_options)
        else:
            return None
    
    def _filter_options(self, options: List[Dict], current_time: pd.Timestamp) -> List[Strike]:
        """Filter options by expiry and liquidity requirements."""
        valid_options = []
        
        for opt in options:
            # Check expiry
            expiry = pd.to_datetime(opt['expiry'])
            days_to_expiry = (expiry - current_time).days
            
            if (days_to_expiry < self.config.market.min_days_to_expiry or 
                days_to_expiry > self.config.market.max_days_to_expiry):
                continue
            
            # Check liquidity
            if (opt.get('open_interest', 0) < self.config.market.min_open_interest or
                opt.get('volume', 0) < 100):  # Minimum volume
                continue
            
            # Check bid-ask spread
            bid = opt.get('bid', 0)
            ask = opt.get('ask', 0)
            if bid <= 0 or ask <= 0:
                continue
            
            spread_pct = (ask - bid) / ((ask + bid) / 2)
            if spread_pct > self.config.market.max_bid_ask_spread_pct:
                continue
            
            # Create Strike object
            strike = Strike(
                strike_price=opt['strike'],
                option_type=OptionType(opt['option_type']),
                expiry_date=expiry,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2,
                open_interest=opt.get('open_interest', 0),
                volume=opt.get('volume', 0),
                implied_vol=opt.get('implied_vol', 0.2),
                delta=opt.get('delta'),
                gamma=opt.get('gamma'),
                theta=opt.get('theta'),
                vega=opt.get('vega')
            )
            
            valid_options.append(strike)
        
        return valid_options
    
    def _select_bull_call_strikes(self, 
                                 current_price: float,
                                 expected_return: float,
                                 options: List[Strike]) -> Optional[Tuple[Strike, Strike]]:
        """Select strikes for bull call spread."""
        # Filter for call options
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        if len(calls) < 2:
            return None
        
        # Calculate forecast band
        band_coeff = np.random.uniform(*self.config.structure.band_coefficient_range)
        forecast_vol = 0.2  # Default volatility
        band_width = band_coeff * forecast_vol * np.sqrt(self.config.market.trading_horizon_days / 252)
        
        lower_bound = current_price * (1 + expected_return - band_width)
        upper_bound = current_price * (1 + expected_return + band_width)
        
        # Select long strike (buy) - ATM or slightly ITM
        long_strikes = [opt for opt in calls if abs(opt.strike_price - current_price) / current_price < self.config.structure.atm_strike_tolerance_pct]
        if not long_strikes:
            # If no ATM options, find closest
            long_strikes = sorted(calls, key=lambda x: abs(x.strike_price - current_price))[:5]
        
        long_strike = min(long_strikes, key=lambda x: x.strike_price)  # Choose lowest strike
        
        # Select short strike (sell) - near upper bound
        target_short_strike = upper_bound
        short_strikes = [opt for opt in calls if opt.strike_price > long_strike.strike_price]
        if not short_strikes:
            return None
        
        short_strike = min(short_strikes, key=lambda x: abs(x.strike_price - target_short_strike))
        
        return long_strike, short_strike
    
    def _select_bear_put_strikes(self, 
                                current_price: float,
                                expected_return: float,
                                options: List[Strike]) -> Optional[Tuple[Strike, Strike]]:
        """Select strikes for bear put spread."""
        # Filter for put options
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        if len(puts) < 2:
            return None
        
        # Calculate forecast band
        band_coeff = np.random.uniform(*self.config.structure.band_coefficient_range)
        forecast_vol = 0.2  # Default volatility
        band_width = band_coeff * forecast_vol * np.sqrt(self.config.market.trading_horizon_days / 252)
        
        lower_bound = current_price * (1 + expected_return - band_width)
        upper_bound = current_price * (1 + expected_return + band_width)
        
        # Select long strike (buy) - ATM or slightly ITM
        long_strikes = [opt for opt in puts if abs(opt.strike_price - current_price) / current_price < self.config.structure.atm_strike_tolerance_pct]
        if not long_strikes:
            # If no ATM options, find closest
            long_strikes = sorted(puts, key=lambda x: abs(x.strike_price - current_price))[:5]
        
        long_strike = max(long_strikes, key=lambda x: x.strike_price)  # Choose highest strike
        
        # Select short strike (sell) - near lower bound
        target_short_strike = lower_bound
        short_strikes = [opt for opt in puts if opt.strike_price < long_strike.strike_price]
        if not short_strikes:
            return None
        
        short_strike = max(short_strikes, key=lambda x: abs(x.strike_price - target_short_strike))
        
        return long_strike, short_strike
    
    def _create_structure(self, 
                         structure_type: StructureType,
                         strikes: Tuple[Strike, Strike],
                         current_price: float) -> OptionStructure:
        """Create option structure from selected strikes."""
        long_strike, short_strike = strikes
        
        # Calculate structure metrics
        if structure_type == StructureType.BULL_CALL_SPREAD:
            net_premium = short_strike.mid - long_strike.mid  # Receive premium
            max_profit = short_strike.strike_price - long_strike.strike_price - net_premium
            max_loss = net_premium
            breakeven = long_strike.strike_price + net_premium
            width = short_strike.strike_price - long_strike.strike_price
            
        elif structure_type == StructureType.BEAR_PUT_SPREAD:
            net_premium = long_strike.mid - short_strike.mid  # Pay premium
            max_profit = long_strike.strike_price - short_strike.strike_price - net_premium
            max_loss = net_premium
            breakeven = long_strike.strike_price - net_premium
            width = long_strike.strike_price - short_strike.strike_price
        
        else:
            raise ValueError(f"Unsupported structure type: {structure_type}")
        
        return OptionStructure(
            structure_type=structure_type,
            long_leg=long_strike,
            short_leg=short_strike,
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            width=width
        )
    
    def _check_liquidity(self, structure: OptionStructure, market_data: Dict[str, Any]) -> bool:
        """Check if structure meets liquidity requirements."""
        # Check individual leg liquidity
        for leg in [structure.long_leg, structure.short_leg]:
            if leg.open_interest < self.config.market.min_open_interest:
                return False
            
            if leg.volume < 100:  # Minimum volume
                return False
            
            # Check bid-ask spread
            spread_pct = (leg.ask - leg.bid) / leg.mid
            if spread_pct > self.config.market.max_bid_ask_spread_pct:
                return False
        
        # Check structure-level requirements
        if structure.width < current_price * 0.01:  # Minimum 1% width
            return False
        
        if abs(structure.net_premium) < 0.01:  # Minimum premium
            return False
        
        return True
    
    def calculate_expected_value(self, 
                               structure: OptionStructure,
                               forecast: Dict[str, Any],
                               current_price: float) -> float:
        """Calculate expected value of the structure under P-measure."""
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
        if structure.structure_type == StructureType.BULL_CALL_SPREAD:
            payoff = max(0, min(expected_price - structure.long_leg.strike_price, structure.width)) - structure.net_premium
        elif structure.structure_type == StructureType.BEAR_PUT_SPREAD:
            payoff = max(0, min(structure.long_leg.strike_price - expected_price, structure.width)) - structure.net_premium
        else:
            payoff = 0
        
        return payoff
    
    def calculate_greeks(self, structure: OptionStructure) -> Dict[str, float]:
        """Calculate net Greeks for the structure."""
        greeks = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            long_greek = getattr(structure.long_leg, greek, 0) or 0
            short_greek = getattr(structure.short_leg, greek, 0) or 0
            greeks[greek] = long_greek - short_greek
        
        return greeks
    
    def validate_structure(self, structure: OptionStructure) -> Tuple[bool, str]:
        """Validate structure for risk management."""
        # Check maximum loss
        if structure.max_loss > current_price * self.config.risk.max_loss_per_trade:
            return False, f"Maximum loss {structure.max_loss:.2f} exceeds limit"
        
        # Check width
        if structure.width < current_price * 0.01:
            return False, "Structure width too small"
        
        # Check premium
        if abs(structure.net_premium) < 0.01:
            return False, "Premium too small"
        
        # Check Greeks (if available)
        greeks = self.calculate_greeks(structure)
        
        if abs(greeks.get('delta', 0)) > self.config.risk.max_delta_exposure:
            return False, f"Delta exposure {greeks['delta']:.3f} exceeds limit"
        
        if abs(greeks.get('gamma', 0)) > self.config.risk.max_gamma_exposure:
            return False, f"Gamma exposure {greeks['gamma']:.3f} exceeds limit"
        
        return True, "Structure validation passed"
    
    def get_structure_summary(self, structure: OptionStructure) -> Dict[str, Any]:
        """Get summary of structure for reporting."""
        greeks = self.calculate_greeks(structure)
        
        return {
            'structure_type': structure.structure_type.value,
            'long_strike': structure.long_leg.strike_price,
            'short_strike': structure.short_leg.strike_price,
            'net_premium': structure.net_premium,
            'max_profit': structure.max_profit,
            'max_loss': structure.max_loss,
            'breakeven': structure.breakeven,
            'width': structure.width,
            'width_pct': structure.width / structure.long_leg.strike_price,
            'greeks': greeks,
            'liquidity': {
                'long_oi': structure.long_leg.open_interest,
                'short_oi': structure.short_leg.open_interest,
                'long_volume': structure.long_leg.volume,
                'short_volume': structure.short_leg.volume
            }
        }
