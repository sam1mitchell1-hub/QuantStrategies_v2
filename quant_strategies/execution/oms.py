"""
Order Management System (OMS)

Orchestrates the complete order flow:
Signal → Validation → Sizing → Order Creation → Execution → Position Update

Main entry point for processing trading signals.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import logging

from .models import Signal, Order, OrderStatus, OrderType, SignalType
from .portfolio import Portfolio
from .ledger import Ledger
from .broker_agent import BrokerAgent
from .sizing import PositionSizer

logger = logging.getLogger(__name__)


class OrderManagementSystem:
    """
    Order Management System - orchestrates signal processing to position updates.
    
    Flow:
    1. Receive signal from strategy
    2. Check portfolio constraints
    3. Calculate position size
    4. Create order
    5. Execute via broker
    6. Record fills
    7. Update portfolio
    8. Log everything to ledger
    """
    
    def __init__(self,
                 portfolio: Portfolio,
                 ledger: Ledger,
                 broker: BrokerAgent,
                 sizer: PositionSizer):
        """
        Initialize OMS.
        
        Args:
            portfolio: Portfolio manager
            ledger: Order ledger
            broker: Broker agent
            sizer: Position sizer
        """
        self.portfolio = portfolio
        self.ledger = ledger
        self.broker = broker
        self.sizer = sizer
        
        logger.info("OrderManagementSystem initialized")
    
    def process_signal(self, signal: Signal, 
                      current_market_data: Dict[str, Any]) -> Optional[Order]:
        """
        Process a trading signal through complete order flow.
        
        Args:
            signal: Trading signal to process
            current_market_data: Current market data for the ticker
            
        Returns:
            Order object with final status, or None if rejected
        """
        logger.info(f"Processing signal: {signal.signal_id} - {signal.ticker} {signal.signal_type.value}")
        
        # Record signal to ledger
        self.ledger.record_signal(signal)
        
        # Determine current price
        current_price = self._get_current_price(signal.ticker, current_market_data)
        if current_price is None or current_price <= 0:
            logger.error(f"Invalid price for {signal.ticker}: {current_price}")
            return None
        
        # Route to appropriate handler
        if signal.signal_type == SignalType.BUY:
            return self._process_buy_signal(signal, current_price, current_market_data)
        elif signal.signal_type == SignalType.SELL:
            return self._process_sell_signal(signal, current_price, current_market_data)
        else:
            logger.error(f"Unknown signal type: {signal.signal_type}")
            return None
    
    def _process_buy_signal(self, signal: Signal, current_price: float,
                           market_data: Dict[str, Any]) -> Optional[Order]:
        """
        Process a BUY signal.
        
        Args:
            signal: Buy signal
            current_price: Current price
            market_data: Market data
            
        Returns:
            Order object or None if rejected
        """
        # Check if position already exists
        if self.portfolio.has_position(signal.ticker):
            logger.info(f"Rejecting BUY signal: position already exists for {signal.ticker}")
            return None
        
        # Calculate position size
        quantity = self.sizer.calculate_size(self.portfolio, signal.ticker, current_price)
        
        if quantity <= 0:
            logger.info(f"Rejecting BUY signal: position size is 0 for {signal.ticker}")
            return None
        
        # Check portfolio constraints
        is_valid, reason = self.portfolio.check_constraints(signal.ticker, quantity, current_price)
        if not is_valid:
            logger.info(f"Rejecting BUY signal: {reason}")
            return None
        
        # Create order
        order = Order.create(signal, quantity, OrderType.MARKET)
        logger.info(f"Created order: {order.order_id} - BUY {quantity} {signal.ticker} @ ~${current_price:.2f}")
        
        # Record order to ledger
        self.ledger.record_order(order)
        
        # Submit order
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now()
        self.ledger.record_order(order)
        
        # Execute order
        fills = self.broker.execute_order(order, market_data)
        
        if not fills:
            # Execution failed
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            self.ledger.record_order(order)
            logger.error(f"Order execution failed: {order.order_id}")
            return order
        
        # Process fills
        total_filled = 0
        for fill in fills:
            # Record fill
            self.ledger.record_fill(fill)
            
            # Update portfolio
            self.portfolio.update_position(signal.ticker, fill.quantity, fill.price)
            
            # Update cash for fees
            self.portfolio.cash -= fill.fees
            
            total_filled += fill.quantity
            logger.info(f"Processed fill: {fill.fill_id} - {fill.quantity} @ ${fill.price:.2f}")
        
        # Update order status
        order.filled_quantity = total_filled
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILL
        order.updated_at = datetime.now()
        self.ledger.record_order(order)
        
        # Record position snapshot
        position = self.portfolio.get_position(signal.ticker)
        if position:
            self.ledger.record_position_snapshot(
                signal.ticker,
                position.quantity,
                position.avg_cost,
                position.market_value,
                position.unrealized_pnl
            )
        
        logger.info(f"Order completed: {order.order_id} - {order.status.value}")
        return order
    
    def _process_sell_signal(self, signal: Signal, current_price: float,
                            market_data: Dict[str, Any]) -> Optional[Order]:
        """
        Process a SELL signal (close position).
        
        Args:
            signal: Sell signal
            current_price: Current price
            market_data: Market data
            
        Returns:
            Order object or None if rejected
        """
        # Check if position exists
        if not self.portfolio.has_position(signal.ticker):
            logger.info(f"Rejecting SELL signal: no position exists for {signal.ticker}")
            return None
        
        # Calculate exit size (full position close)
        quantity = self.sizer.calculate_exit_size(self.portfolio, signal.ticker)
        
        if quantity >= 0:  # Should be negative
            logger.error(f"Invalid exit size for {signal.ticker}: {quantity}")
            return None
        
        # Check portfolio constraints
        is_valid, reason = self.portfolio.check_constraints(signal.ticker, quantity, current_price)
        if not is_valid:
            logger.info(f"Rejecting SELL signal: {reason}")
            return None
        
        # Create order (quantity is negative for sell)
        order = Order.create(signal, abs(quantity), OrderType.MARKET)
        logger.info(f"Created order: {order.order_id} - SELL {abs(quantity)} {signal.ticker} @ ~${current_price:.2f}")
        
        # Record order to ledger
        self.ledger.record_order(order)
        
        # Submit order
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now()
        self.ledger.record_order(order)
        
        # Execute order
        fills = self.broker.execute_order(order, market_data)
        
        if not fills:
            # Execution failed
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            self.ledger.record_order(order)
            logger.error(f"Order execution failed: {order.order_id}")
            return order
        
        # Process fills (sells reduce position, so negate quantity)
        total_filled = 0
        for fill in fills:
            # Record fill
            self.ledger.record_fill(fill)
            
            # Update portfolio (negative quantity for sell)
            self.portfolio.update_position(signal.ticker, -fill.quantity, fill.price)
            
            # Update cash for fees
            self.portfolio.cash -= fill.fees
            
            total_filled += fill.quantity
            logger.info(f"Processed fill: {fill.fill_id} - {fill.quantity} @ ${fill.price:.2f}")
        
        # Update order status
        order.filled_quantity = total_filled
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILL
        order.updated_at = datetime.now()
        self.ledger.record_order(order)
        
        # Record position snapshot
        position = self.portfolio.get_position(signal.ticker)
        if position:
            self.ledger.record_position_snapshot(
                signal.ticker,
                position.quantity,
                position.avg_cost,
                position.market_value,
                position.unrealized_pnl
            )
        
        logger.info(f"Order completed: {order.order_id} - {order.status.value}")
        return order
    
    def _get_current_price(self, ticker: str, 
                          market_data: Dict[str, Any]) -> Optional[float]:
        """
        Extract current price from market data.
        
        Args:
            ticker: Stock ticker
            market_data: Market data dict
            
        Returns:
            Current price or None
        """
        # Try various price fields
        for field in ['last', 'close', 'mid', 'ask', 'bid']:
            price = market_data.get(field)
            if price is not None and price > 0:
                return float(price)
        
        logger.error(f"Cannot determine current price for {ticker} from market data: {market_data}")
        return None
    
    def get_portfolio_status(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """
        Get current portfolio status.
        
        Args:
            current_prices: Optional dict of current prices
            
        Returns:
            Portfolio summary dict
        """
        return self.portfolio.get_portfolio_summary(current_prices)
    
    def print_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None):
        """
        Print portfolio summary.
        
        Args:
            current_prices: Optional dict of current prices
        """
        self.portfolio.print_summary(current_prices)

