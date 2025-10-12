"""
Broker Agent

Handles order execution via broker API.
Currently a placeholder that simulates instant execution at bid/ask prices.

Future: Integration with Interactive Brokers API.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .models import Order, Fill, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class BrokerAgent:
    """
    Agent for executing orders via broker.
    
    Current implementation: Placeholder with instant execution at bid/ask.
    Future: Interactive Brokers API integration.
    """
    
    def __init__(self, 
                 commission_per_share: float = 0.005,
                 min_commission: float = 1.0,
                 slippage_bps: float = 5.0):
        """
        Initialize broker agent.
        
        Args:
            commission_per_share: Commission per share (default $0.005)
            min_commission: Minimum commission per order (default $1.00)
            slippage_bps: Slippage in basis points (default 5 bps)
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        
        logger.info(f"BrokerAgent initialized (SIMULATION MODE): "
                   f"commission=${commission_per_share}/share, "
                   f"min=${min_commission}, slippage={slippage_bps}bps")
    
    def execute_order(self, order: Order, 
                     current_market_data: Dict[str, Any]) -> List[Fill]:
        """
        Execute an order and return fills.
        
        Current implementation: Instant fill at bid/ask with simulated costs.
        
        Args:
            order: Order to execute
            current_market_data: Market data dict with 'bid', 'ask', 'last', etc.
            
        Returns:
            List of Fill objects (currently always 1 fill with full quantity)
        """
        logger.info(f"Executing order: {order.order_id} - {order.ticker} {order.side.value} {order.quantity}")
        
        # Determine execution price
        execution_price = self._get_execution_price(order, current_market_data)
        
        if execution_price is None:
            logger.error(f"Cannot determine execution price for {order.ticker}")
            return []
        
        # Calculate fees
        fees = self._calculate_fees(order.quantity, execution_price)
        
        # Create fill
        fill = Fill.create(
            order=order,
            quantity=order.quantity,
            price=execution_price,
            fees=fees
        )
        
        logger.info(f"Order filled: {fill.fill_id} - {fill.quantity} @ ${fill.price:.2f}, fees=${fees:.2f}")
        
        return [fill]
    
    def _get_execution_price(self, order: Order, 
                            market_data: Dict[str, Any]) -> Optional[float]:
        """
        Determine execution price based on order side and market data.
        
        Args:
            order: Order being executed
            market_data: Market data dict
            
        Returns:
            Execution price or None if cannot determine
        """
        # Try to get bid/ask from market data
        bid = market_data.get('bid')
        ask = market_data.get('ask')
        last = market_data.get('last')
        close = market_data.get('close')
        
        # Determine base price
        if order.side == OrderSide.BUY:
            # Buy at ask (or last/close with slippage)
            if ask is not None and ask > 0:
                base_price = ask
            elif last is not None and last > 0:
                base_price = last
            elif close is not None and close > 0:
                base_price = close
            else:
                return None
            
            # Add slippage for buys
            execution_price = base_price * (1 + self.slippage_bps / 10000)
            
        else:  # SELL
            # Sell at bid (or last/close with slippage)
            if bid is not None and bid > 0:
                base_price = bid
            elif last is not None and last > 0:
                base_price = last
            elif close is not None and close > 0:
                base_price = close
            else:
                return None
            
            # Subtract slippage for sells
            execution_price = base_price * (1 - self.slippage_bps / 10000)
        
        logger.debug(f"Execution price for {order.ticker} {order.side.value}: "
                    f"${base_price:.2f} → ${execution_price:.2f} "
                    f"(slippage={self.slippage_bps}bps)")
        
        return execution_price
    
    def _calculate_fees(self, quantity: int, price: float) -> float:
        """
        Calculate total fees for a trade.
        
        Args:
            quantity: Number of shares
            price: Price per share
            
        Returns:
            Total fees
        """
        # Commission
        commission = max(
            abs(quantity) * self.commission_per_share,
            self.min_commission
        )
        
        # Could add other fees here (exchange fees, regulatory fees, etc.)
        # For now, just commission
        total_fees = commission
        
        logger.debug(f"Fees calculated: {quantity} shares @ ${price:.2f} = ${total_fees:.2f}")
        
        return total_fees
    
    def get_quote(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get current quote for a ticker.
        
        This is a placeholder - in real implementation would query broker API.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with bid, ask, last, etc. or None
        """
        logger.warning(f"get_quote() not implemented - placeholder mode")
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        This is a placeholder - in real implementation would cancel via broker API.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        logger.warning(f"cancel_order() not implemented - placeholder mode")
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current status of an order.
        
        This is a placeholder - in real implementation would query broker API.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            OrderStatus or None
        """
        logger.warning(f"get_order_status() not implemented - placeholder mode")
        return None

