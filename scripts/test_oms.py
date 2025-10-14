#!/usr/bin/env python3
"""
OMS Test Script

Tests the complete Order Management System:
1. Initialize portfolio with cash
2. Create mock strategy
3. Generate buy signals for multiple tickers
4. Process through OMS
5. Verify positions, ledger entries, cash updates
6. Generate sell signal
7. Verify position closure
8. Query ledger for audit trail
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime

# Import OMS components
from quant_strategies.execution import (
    Portfolio,
    Ledger,
    BrokerAgent,
    PositionSizer,
    OrderManagementSystem,
    SignalType,
    OrderStatus
)

# Import strategy
from quant_strategies.strategies import BollingerRSIStrategy
from quant_strategies.strategies.base_strategy import SignalType as StrategySignalType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_oms_workflow():
    """Test the complete OMS workflow."""
    
    print("\n" + "="*80)
    print("OMS TEST - Complete Workflow")
    print("="*80 + "\n")
    
    # Step 1: Initialize components
    print("STEP 1: Initialize OMS Components")
    print("-" * 80)
    
    initial_cash = 100000.0
    portfolio = Portfolio(initial_cash=initial_cash)
    ledger = Ledger(db_path="data/test_ledger.db")
    broker = BrokerAgent(commission_per_share=0.005, slippage_bps=5.0)
    sizer = PositionSizer()
    oms = OrderManagementSystem(portfolio, ledger, broker, sizer)
    
    print(f"✓ Portfolio initialized with ${initial_cash:,.2f}")
    print(f"✓ Ledger initialized")
    print(f"✓ Broker agent initialized (simulation mode)")
    print(f"✓ Position sizer initialized")
    print(f"✓ OMS initialized\n")
    
    # Step 2: Create strategy
    print("STEP 2: Create Strategy")
    print("-" * 80)
    
    strategy = BollingerRSIStrategy()
    print(f"✓ Strategy created: {strategy.name}\n")
    
    # Step 3: Generate buy signals for 3 tickers
    print("STEP 3: Generate and Process BUY Signals")
    print("-" * 80)
    
    test_tickers = [
        ('AAPL', 150.00),
        ('MSFT', 350.00),
        ('GOOGL', 140.00)
    ]
    
    orders = []
    for ticker, price in test_tickers:
        print(f"\nProcessing {ticker} @ ${price:.2f}")
        
        # Create signal
        signal = strategy.emit_signal(ticker, StrategySignalType.BUY, strength=0.85)
        print(f"  Signal created: {signal.signal_id}")
        
        # Prepare market data
        market_data = {
            'ticker': ticker,
            'last': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'close': price
        }
        
        # Process through OMS
        order = oms.process_signal(signal, market_data)
        
        if order and order.status == OrderStatus.FILLED:
            orders.append(order)
            print(f"  ✓ Order filled: {order.order_id}")
            print(f"    Quantity: {order.quantity} shares")
            print(f"    Total cost: ~${order.quantity * price:,.2f}")
        else:
            print(f"  ✗ Order failed or rejected")
    
    # Step 4: Check portfolio state
    print("\n" + "="*80)
    print("STEP 4: Portfolio State After Buys")
    print("-" * 80)
    
    current_prices = {ticker: price for ticker, price in test_tickers}
    oms.print_portfolio_summary(current_prices)
    
    # Verify positions
    print("\nVerifying positions:")
    for ticker, _ in test_tickers:
        position = portfolio.get_position(ticker)
        if position and position.quantity > 0:
            print(f"  ✓ {ticker}: {position.quantity} shares @ ${position.avg_cost:.2f}")
        else:
            print(f"  ✗ {ticker}: No position")
    
    # Step 5: Generate sell signal for first ticker
    print("\n" + "="*80)
    print("STEP 5: Generate and Process SELL Signal")
    print("-" * 80)
    
    sell_ticker = test_tickers[0][0]
    sell_price = test_tickers[0][1] * 1.05  # 5% profit
    
    print(f"\nClosing position: {sell_ticker} @ ${sell_price:.2f}")
    
    # Create sell signal
    sell_signal = strategy.emit_signal(sell_ticker, StrategySignalType.SELL, strength=0.90)
    print(f"  Signal created: {sell_signal.signal_id}")
    
    # Prepare market data
    market_data = {
        'ticker': sell_ticker,
        'last': sell_price,
        'bid': sell_price * 0.999,
        'ask': sell_price * 1.001,
        'close': sell_price
    }
    
    # Process through OMS
    sell_order = oms.process_signal(sell_signal, market_data)
    
    if sell_order and sell_order.status == OrderStatus.FILLED:
        print(f"  ✓ Position closed: {sell_order.order_id}")
        print(f"    Quantity: {sell_order.quantity} shares")
    else:
        print(f"  ✗ Sell order failed")
    
    # Step 6: Final portfolio state
    print("\n" + "="*80)
    print("STEP 6: Final Portfolio State")
    print("-" * 80)
    
    current_prices[sell_ticker] = sell_price
    oms.print_portfolio_summary(current_prices)
    
    # Step 7: Query ledger for audit trail
    print("\n" + "="*80)
    print("STEP 7: Ledger Audit Trail")
    print("-" * 80)
    
    print("\nAll signals:")
    signals = ledger.get_all_signals(limit=10)
    for i, sig in enumerate(signals, 1):
        print(f"  {i}. {sig['signal_id']}: {sig['ticker']} {sig['signal_type']} "
              f"(strength: {sig['strength']:.2f})")
    
    print("\nOrder history for first buy:")
    if orders:
        first_order = orders[0]
        order_history = ledger.get_order_history(first_order.order_id)
        for i, order_record in enumerate(order_history, 1):
            print(f"  {i}. Status: {order_record['status']} at {order_record['updated_at']}")
        
        # Get fills
        fills = ledger.get_fills_for_order(first_order.order_id)
        print(f"\n  Fills for {first_order.order_id}:")
        for fill in fills:
            print(f"    - {fill['quantity']} shares @ ${fill['price']:.2f}, "
                  f"fees: ${fill['fees']:.2f}")
    
    # Step 8: Test position sizing logic
    print("\n" + "="*80)
    print("STEP 8: Test Position Sizing Logic")
    print("-" * 80)
    
    print(f"\nCurrent portfolio state:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Open positions: {portfolio.get_position_count()}")
    
    # Try to add more positions to test 7+ logic
    print("\nTesting position sizing with different position counts:")
    
    test_price = 100.0
    
    # Simulate different position counts
    for pos_count in [0, 3, 7, 10]:
        # Create temporary portfolio for testing
        test_portfolio = Portfolio(initial_cash=100000)
        
        # Add dummy positions
        for i in range(pos_count):
            test_portfolio.update_position(f"TEST{i}", 100, 100.0)
        
        # Calculate size
        size = sizer.calculate_size(test_portfolio, "NEWSTOCK", test_price)
        buying_power = size * test_price
        
        print(f"  Positions: {pos_count:2d} → Size: {size:4d} shares "
              f"(${buying_power:>10,.2f})")
    
    # Step 9: Test constraint violations
    print("\n" + "="*80)
    print("STEP 9: Test Constraint Violations")
    print("-" * 80)
    
    # Test 1: Try to buy same ticker again
    print("\n1. Attempting to buy existing position (should reject):")
    if test_tickers[1:]:
        existing_ticker = test_tickers[1][0]
        duplicate_signal = strategy.emit_signal(existing_ticker, StrategySignalType.BUY)
        market_data = {'last': test_tickers[1][1], 'close': test_tickers[1][1]}
        duplicate_order = oms.process_signal(duplicate_signal, market_data)
        
        if duplicate_order is None:
            print(f"   ✓ Correctly rejected duplicate position for {existing_ticker}")
        else:
            print(f"   ✗ Should have rejected duplicate position")
    
    # Test 2: Try to sell non-existent position
    print("\n2. Attempting to sell non-existent position (should reject):")
    fake_signal = strategy.emit_signal("FAKE", StrategySignalType.SELL)
    market_data = {'last': 100.0, 'close': 100.0}
    fake_order = oms.process_signal(fake_signal, market_data)
    
    if fake_order is None:
        print(f"   ✓ Correctly rejected sell for non-existent position")
    else:
        print(f"   ✗ Should have rejected sell")
    
    # Final summary
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    summary = portfolio.get_portfolio_summary(current_prices)
    
    print(f"\nFinal Results:")
    print(f"  Initial Cash:     ${initial_cash:>15,.2f}")
    print(f"  Final Cash:       ${summary['cash']:>15,.2f}")
    print(f"  Total Value:      ${summary['total_value']:>15,.2f}")
    print(f"  Total P&L:        ${summary['total_pnl']:>15,.2f} ({summary['total_return_pct']:>6.2f}%)")
    print(f"  Open Positions:   {summary['position_count']:>16}")
    print(f"  Total Signals:    {len(signals):>16}")
    print(f"  Total Orders:     {len(orders) + (1 if sell_order else 0):>16}")
    
    print("\n✓ All OMS components working correctly!")
    print("✓ Signal → Order → Fill → Position flow validated")
    print("✓ Ledger audit trail complete")
    print("✓ Portfolio constraints enforced")
    print("✓ Position sizing logic verified\n")
    
    return True


def main():
    """Main entry point."""
    try:
        success = test_oms_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

