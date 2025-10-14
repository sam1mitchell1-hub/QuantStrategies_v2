"""
Position Sizing

Dynamic position sizing based on portfolio state and constraints.

Rules:
- 0 positions: 10% of total cash
- 1-7 positions: 10% of remaining cash
- 7+ positions: size to keep 20% cash buffer
- Max 1 position per ticker (return 0 if position exists)
"""

from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates position sizes based on dynamic portfolio rules.
    """
    
    def __init__(self, 
                 initial_position_pct: float = 0.10,
                 ongoing_position_pct: float = 0.10,
                 max_positions_before_buffer: int = 7,
                 min_cash_buffer_pct: float = 0.20):
        """
        Initialize position sizer.
        
        Args:
            initial_position_pct: Position size % when portfolio is empty (default 10%)
            ongoing_position_pct: Position size % for ongoing positions (default 10%)
            max_positions_before_buffer: Threshold for applying cash buffer (default 7)
            min_cash_buffer_pct: Minimum cash buffer % when above threshold (default 20%)
        """
        self.initial_position_pct = initial_position_pct
        self.ongoing_position_pct = ongoing_position_pct
        self.max_positions_before_buffer = max_positions_before_buffer
        self.min_cash_buffer_pct = min_cash_buffer_pct
        
        logger.info(f"PositionSizer initialized: {initial_position_pct:.0%} initial, "
                   f"{ongoing_position_pct:.0%} ongoing, buffer at {max_positions_before_buffer}+ positions")
    
    def calculate_size(self, portfolio: 'Portfolio', ticker: str, 
                      current_price: float) -> int:
        """
        Calculate position size for a trade.
        
        Args:
            portfolio: Portfolio object
            ticker: Stock ticker
            current_price: Current price per share
            
        Returns:
            Number of shares to buy (0 if position already exists or constraints violated)
        """
        # Rule 1: Max 1 position per ticker
        if portfolio.has_position(ticker):
            logger.info(f"Position already exists for {ticker}, size = 0")
            return 0
        
        # Get portfolio state
        position_count = portfolio.get_position_count()
        available_cash = portfolio.get_available_cash()
        total_cash = portfolio.cash
        
        # Calculate buying power based on rules
        if position_count == 0:
            # Rule 2: First position = 10% of total cash
            buying_power = total_cash * self.initial_position_pct
            logger.debug(f"First position: {self.initial_position_pct:.0%} of ${total_cash:,.2f} = ${buying_power:,.2f}")
            
        elif position_count < self.max_positions_before_buffer:
            # Rule 3: Positions 1-7 = 10% of remaining cash
            buying_power = available_cash * self.ongoing_position_pct
            logger.debug(f"Position #{position_count + 1}: {self.ongoing_position_pct:.0%} of ${available_cash:,.2f} = ${buying_power:,.2f}")
            
        else:
            # Rule 4: 7+ positions = size to keep 20% cash buffer
            # Calculate total portfolio value
            position_value = sum(
                pos.quantity * pos.avg_cost 
                for pos in portfolio.positions.values() 
                if pos.quantity != 0
            )
            total_value = total_cash + position_value
            
            # Calculate maximum position size that maintains 20% cash buffer
            min_cash_required = total_value * self.min_cash_buffer_pct
            max_position_value = max(0, available_cash - min_cash_required)
            
            # Also limit to 10% of available cash
            buying_power = min(
                available_cash * self.ongoing_position_pct,
                max_position_value
            )
            
            logger.debug(f"Position #{position_count + 1} (7+): "
                        f"min_cash=${min_cash_required:,.2f}, "
                        f"max_position=${max_position_value:,.2f}, "
                        f"buying_power=${buying_power:,.2f}")
        
        # Convert to shares
        if current_price <= 0:
            logger.error(f"Invalid price for {ticker}: ${current_price}")
            return 0
        
        shares = int(buying_power / current_price)
        
        # Ensure we have at least 1 share if we have buying power
        if buying_power > current_price and shares == 0:
            shares = 1
        
        logger.info(f"Calculated size for {ticker}: {shares} shares @ ${current_price:.2f} = ${shares * current_price:,.2f}")
        
        return shares
    
    def calculate_exit_size(self, portfolio: 'Portfolio', ticker: str) -> int:
        """
        Calculate exit size (close entire position).
        
        Args:
            portfolio: Portfolio object
            ticker: Stock ticker
            
        Returns:
            Number of shares to sell (negative) or 0 if no position
        """
        position = portfolio.get_position(ticker)
        
        if not position or position.quantity == 0:
            logger.info(f"No position to close for {ticker}")
            return 0
        
        # Return negative quantity to indicate sell
        exit_size = -position.quantity
        logger.info(f"Exit size for {ticker}: {exit_size} shares (close position)")
        
        return exit_size

