"""
Portfolio Management

Tracks cash and positions with real-time valuation and P&L.
Enforces position limits and risk constraints.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from .models import Position

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio manager for tracking cash and positions.
    
    Maintains:
    - Cash balance
    - Open positions
    - Portfolio value
    - P&L calculations
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0  # Track actual profits/losses from closed trades
        self.created_at = datetime.now()
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f}")
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """
        Get position for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Position object or None if no position
        """
        return self.positions.get(ticker)
    
    def has_position(self, ticker: str) -> bool:
        """
        Check if portfolio has a position in ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            True if position exists with non-zero quantity
        """
        position = self.positions.get(ticker)
        return position is not None and position.quantity != 0
    
    def get_position_count(self) -> int:
        """
        Get number of open positions.
        
        Returns:
            Count of positions with non-zero quantity
        """
        return sum(1 for pos in self.positions.values() if pos.quantity != 0)
    
    def get_available_cash(self) -> float:
        """
        Get cash available for trading.
        
        Returns:
            Available cash balance
        """
        return self.cash
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dict of {ticker: current_price}
            
        Returns:
            Total portfolio value (cash + positions)
        """
        # Update position values
        for ticker, position in self.positions.items():
            if ticker in current_prices and position.quantity != 0:
                position.update_market_value(current_prices[ticker])
        
        # Sum up all values
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    def calculate_pnl(self, current_prices: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate portfolio P&L.
        
        Args:
            current_prices: Dict of {ticker: current_price}
            
        Returns:
            Tuple of (unrealized_pnl, realized_pnl, total_pnl)
        """
        # Update position values and calculate unrealized P&L
        unrealized_pnl = 0.0
        for ticker, position in self.positions.items():
            if ticker in current_prices and position.quantity != 0:
                position.update_market_value(current_prices[ticker])
                unrealized_pnl += position.unrealized_pnl
        
        # Get total portfolio value
        total_value = self.get_total_value(current_prices)
        
        # Total P&L = current value - starting value
        total_pnl = total_value - self.initial_cash
        
        # Realized P&L comes from closed trades (tracked separately)
        realized_pnl = self.realized_pnl
        
        return unrealized_pnl, realized_pnl, total_pnl
    
    def check_constraints(self, ticker: str, quantity: int, price: float) -> Tuple[bool, str]:
        """
        Check if a trade violates portfolio constraints.
        
        Args:
            ticker: Stock ticker
            side: BUY or SELL
            quantity: Number of shares
            price: Price per share
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check 1: No position exists for this ticker (no doubling down)
        if self.has_position(ticker) and quantity > 0:
            return False, f"Position already exists for {ticker}"
        
        # Check 2: Sufficient cash for buy orders
        if quantity > 0:  # Buy order
            required_cash = quantity * price
            if required_cash > self.cash:
                return False, f"Insufficient cash: need ${required_cash:,.2f}, have ${self.cash:,.2f}"
        
        # Check 3: Have sufficient shares for sell orders
        if quantity < 0:  # Sell order
            position = self.get_position(ticker)
            if not position or position.quantity < abs(quantity):
                have = position.quantity if position else 0
                return False, f"Insufficient shares: trying to sell {abs(quantity)}, have {have}"
        
        # Check 4: Quantity must be positive
        if quantity == 0:
            return False, "Order quantity is zero"
        
        return True, "All constraints satisfied"
    
    def update_position(self, ticker: str, quantity: int, price: float):
        """
        Update or create a position.
        
        Args:
            ticker: Stock ticker
            quantity: Shares to add (positive) or remove (negative)
            price: Transaction price per share
        """
        # Update cash
        cash_change = -quantity * price  # Negative for buys, positive for sells
        self.cash += cash_change
        
        # Update position
        if ticker not in self.positions:
            # Create new position (BUY)
            position = Position(
                ticker=ticker,
                quantity=quantity,
                avg_cost=price,
                market_value=quantity * price,
                unrealized_pnl=0.0
            )
            self.positions[ticker] = position
            logger.info(f"Created new position: {ticker} x{quantity} @ ${price:.2f}")
        else:
            # Update existing position
            position = self.positions[ticker]
            old_quantity = position.quantity
            old_avg_cost = position.avg_cost
            
            # If selling (quantity negative), track realized P&L
            if quantity < 0:
                shares_sold = abs(quantity)
                realized_pnl_this_trade = (price - old_avg_cost) * shares_sold
                self.realized_pnl += realized_pnl_this_trade
                logger.info(f"Realized P&L from sale: ${realized_pnl_this_trade:,.2f} "
                          f"({shares_sold} shares @ ${price:.2f} vs avg ${old_avg_cost:.2f})")
            
            position.add_shares(quantity, price)
            
            logger.info(f"Updated position: {ticker} {old_quantity} → {position.quantity} @ ${price:.2f}")
            
            # Remove position if closed
            if position.quantity == 0:
                logger.info(f"Position closed: {ticker}")
                # Keep the position object but with 0 quantity for history
    
    def get_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """
        Get comprehensive portfolio summary.
        
        Args:
            current_prices: Optional dict of current prices for valuation
            
        Returns:
            Dictionary with portfolio metrics
        """
        if current_prices is None:
            current_prices = {}
        
        # Calculate values
        total_value = self.get_total_value(current_prices)
        unrealized_pnl, realized_pnl, total_pnl = self.calculate_pnl(current_prices)
        
        # Position details
        active_positions = [
            {
                'ticker': ticker,
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current_price': current_prices.get(ticker, 0.0),
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'pnl_pct': (pos.unrealized_pnl / (pos.quantity * pos.avg_cost) * 100) if pos.quantity != 0 else 0
            }
            for ticker, pos in self.positions.items()
            if pos.quantity != 0
        ]
        
        return {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'position_count': self.get_position_count(),
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.initial_cash * 100) if self.initial_cash > 0 else 0,
            'positions': active_positions
        }
    
    def print_summary(self, current_prices: Optional[Dict[str, float]] = None):
        """
        Print a formatted portfolio summary.
        
        Args:
            current_prices: Optional dict of current prices for valuation
        """
        summary = self.get_portfolio_summary(current_prices)
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Cash:              ${summary['cash']:>15,.2f}")
        print(f"Initial Cash:      ${summary['initial_cash']:>15,.2f}")
        print(f"Total Value:       ${summary['total_value']:>15,.2f}")
        print(f"Open Positions:    {summary['position_count']:>16}")
        print("-"*60)
        print(f"Unrealized P&L:    ${summary['unrealized_pnl']:>15,.2f}")
        print(f"Realized P&L:      ${summary['realized_pnl']:>15,.2f}")
        print(f"Total P&L:         ${summary['total_pnl']:>15,.2f} ({summary['total_return_pct']:>6.2f}%)")
        
        if summary['positions']:
            print("-"*75)
            print("POSITIONS:")
            print(f"{'Ticker':<8} {'Qty':>8} {'Avg Cost':>11} {'Curr Price':>11} {'Mkt Value':>13} {'P&L':>13} {'P&L %':>9}")
            print("-"*75)
            for pos in summary['positions']:
                print(f"{pos['ticker']:<8} {pos['quantity']:>8} "
                      f"${pos['avg_cost']:>10,.2f} ${pos['current_price']:>10,.2f} "
                      f"${pos['market_value']:>12,.2f} "
                      f"${pos['unrealized_pnl']:>12,.2f} {pos['pnl_pct']:>8.2f}%")
        
        print("="*75 + "\n")

