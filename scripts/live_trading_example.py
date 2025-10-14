#!/usr/bin/env python3
"""
Live Trading Example

Demonstrates running a strategy with the OMS framework using real stock data.
This is a paper trading simulation - no real money is used.

Usage:
    python scripts/live_trading_example.py [--cycles N] [--interval SECONDS]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import OMS components
from quant_strategies.execution import (
    Portfolio, Ledger, BrokerAgent, PositionSizer, OrderManagementSystem
)

# Import data and strategy components
from quant_strategies.data.live_data_fetcher import LiveDataFetcher
from quant_strategies.strategies import BollingerRSIStrategy
from quant_strategies.strategies.live_strategy_runner import LiveStrategyRunner


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_oms(config: dict) -> OrderManagementSystem:
    """
    Initialize the Order Management System.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized OMS instance
    """
    print("\n" + "="*80)
    print("INITIALIZING ORDER MANAGEMENT SYSTEM")
    print("="*80)
    
    # Create portfolio
    initial_cash = config['portfolio']['initial_cash']
    portfolio = Portfolio(initial_cash=initial_cash)
    print(f"✓ Portfolio created with ${initial_cash:,.2f}")
    
    # Create ledger
    ledger = Ledger(db_path="data/live_trading_ledger.db")
    print(f"✓ Ledger initialized")
    
    # Create broker agent
    broker = BrokerAgent(
        commission_per_share=config['execution']['commission_per_share'],
        min_commission=config['execution']['min_commission'],
        slippage_bps=config['execution']['slippage_bps']
    )
    print(f"✓ Broker agent created (simulation mode)")
    
    # Create position sizer
    sizer = PositionSizer()
    print(f"✓ Position sizer initialized")
    
    # Create OMS
    oms = OrderManagementSystem(portfolio, ledger, broker, sizer)
    print(f"✓ OMS assembled and ready\n")
    
    return oms


def initialize_strategy(config: dict) -> BollingerRSIStrategy:
    """
    Initialize the trading strategy.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Strategy instance
    """
    print("INITIALIZING STRATEGY")
    print("-" * 80)
    
    strategy_params = config.get('strategy', {}).get('parameters', {})
    strategy = BollingerRSIStrategy(parameters=strategy_params)
    
    print(f"✓ Strategy: {strategy.name}")
    print(f"  Parameters: {strategy_params}")
    print()
    
    return strategy


def run_live_trading(config_path: str = "config/live_trading_config.yaml",
                    num_cycles: int = None,
                    cycle_interval: int = None):
    """
    Run live trading simulation.
    
    Args:
        config_path: Path to configuration file
        num_cycles: Number of trading cycles to run (None = infinite)
        cycle_interval: Seconds between cycles (overrides config)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file')
    )
    
    logger = logging.getLogger(__name__)
    
    # Print header
    print("\n" + "="*80)
    print("LIVE TRADING SIMULATION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print()
    
    # Initialize components
    oms = initialize_oms(config)
    strategy = initialize_strategy(config)
    
    # Initialize data fetcher
    print("INITIALIZING DATA FETCHER")
    print("-" * 80)
    data_fetcher = LiveDataFetcher(
        cache_duration_seconds=config['data']['cache_duration_seconds']
    )
    print(f"✓ Data fetcher ready")
    print(f"  Cache duration: {config['data']['cache_duration_seconds']}s")
    print()
    
    # Get watchlist
    watchlist = config.get('watchlist', [])
    print(f"WATCHLIST ({len(watchlist)} tickers):")
    print("-" * 80)
    for ticker in watchlist:
        name = data_fetcher.get_stock_name(ticker)
        print(f"  {ticker:6s} - {name}")
    print()
    
    # Initialize live strategy runner
    print("INITIALIZING LIVE STRATEGY RUNNER")
    print("-" * 80)
    runner = LiveStrategyRunner(
        strategy=strategy,
        oms=oms,
        data_fetcher=data_fetcher,
        watchlist=watchlist,
        bars_for_indicators=config['data']['bars_for_indicators']
    )
    print(f"✓ Strategy runner ready")
    print()
    
    # Determine cycle interval
    interval = cycle_interval or config['schedule']['trading_interval_minutes'] * 60
    
    print("="*80)
    print("STARTING LIVE TRADING")
    print("="*80)
    print(f"Cycle interval: {interval}s ({interval/60:.1f} minutes)")
    print(f"Cycles to run: {'Infinite (Ctrl+C to stop)' if num_cycles is None else num_cycles}")
    print()
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            
            print(f"\n{'='*80}")
            print(f"CYCLE {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # Run trading cycle
            try:
                results = runner.run_cycle()
                
                print(f"\nCycle Results:")
                print(f"  Signals generated: {results['signals_generated']}")
                print(f"  Orders placed:     {results['orders_placed']}")
                print(f"  Orders filled:     {results['orders_filled']}")
                print(f"  Errors:            {results['errors']}")
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)
                print(f"  ✗ Cycle failed: {e}")
            
            # Display portfolio status
            print(f"\n{'='*80}")
            print("PORTFOLIO STATUS")
            print(f"{'='*80}")
            runner.print_portfolio_summary()
            
            # Check if we should stop
            if num_cycles is not None and cycle_count >= num_cycles:
                print(f"\nCompleted {num_cycles} cycles. Stopping.")
                break
            
            # Wait for next cycle
            if num_cycles is None or cycle_count < num_cycles:
                print(f"\nWaiting {interval}s until next cycle...")
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("STOPPING (Keyboard Interrupt)")
        print("="*80)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total cycles run: {cycle_count}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    runner.print_portfolio_summary()
    
    print("\n✓ Live trading simulation complete")
    print(f"✓ Ledger saved to: data/live_trading_ledger.db")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run live trading simulation with OMS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config (infinite cycles)
    python scripts/live_trading_example.py
    
    # Run for 5 cycles
    python scripts/live_trading_example.py --cycles 5
    
    # Run with 30 second intervals
    python scripts/live_trading_example.py --interval 30
    
    # Custom config file
    python scripts/live_trading_example.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/live_trading_config.yaml',
        help='Path to configuration file (default: config/live_trading_config.yaml)'
    )
    
    parser.add_argument(
        '--cycles',
        type=int,
        default=None,
        help='Number of trading cycles to run (default: infinite)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=None,
        help='Seconds between cycles (overrides config file)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run live trading
    run_live_trading(
        config_path=args.config,
        num_cycles=args.cycles,
        cycle_interval=args.interval
    )


if __name__ == "__main__":
    main()

