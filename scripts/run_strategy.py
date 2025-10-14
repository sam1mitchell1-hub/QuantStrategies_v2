#!/usr/bin/env python3
"""
Main Strategy Runner

Orchestrates the complete options trading strategy:
- Loads configuration
- Builds features from market data
- Trains forecasting model
- Runs backtesting
- Generates reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import strategy components
from strategy import (
    StrategyConfig, FeatureBuilder, GBTForecaster, 
    OptionStructureSelector, FDPricer, RiskManager, Backtester
)


def load_sample_data():
    """Load sample market data for testing."""
    print("Loading sample market data...")
    
    # Generate sample price data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate price series with some trend and volatility
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Price series starting at 100
    
    price_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Generate sample options data
    options_data = pd.DataFrame({
        'iv_atm': 0.15 + 0.05 * np.random.randn(len(dates)),
        'iv_skew_90%': 0.20 + 0.05 * np.random.randn(len(dates)),
        'iv_skew_95%': 0.18 + 0.05 * np.random.randn(len(dates)),
        'iv_skew_100%': 0.15 + 0.05 * np.random.randn(len(dates)),
        'iv_skew_105%': 0.17 + 0.05 * np.random.randn(len(dates)),
        'iv_skew_110%': 0.19 + 0.05 * np.random.randn(len(dates)),
        'iv_7d': 0.15 + 0.05 * np.random.randn(len(dates)),
        'iv_14d': 0.16 + 0.05 * np.random.randn(len(dates)),
        'iv_30d': 0.17 + 0.05 * np.random.randn(len(dates)),
        'iv_60d': 0.18 + 0.05 * np.random.randn(len(dates)),
        'iv_90d': 0.19 + 0.05 * np.random.randn(len(dates)),
        'avg_bid_ask_spread': 0.01 + 0.005 * np.random.randn(len(dates)),
        'total_oi': 10000 + 5000 * np.random.randn(len(dates))
    }, index=dates)
    
    # Generate VIX data
    vix_data = pd.DataFrame({
        'vix': 15 + 5 * np.random.randn(len(dates)),
        'vix_9d': 15 + 5 * np.random.randn(len(dates)),
        'vix_30d': 16 + 5 * np.random.randn(len(dates))
    }, index=dates)
    
    # Generate rates data
    rates_data = pd.DataFrame({
        '10y_rate': 0.02 + 0.01 * np.random.randn(len(dates)),
        '2y_rate': 0.015 + 0.01 * np.random.randn(len(dates))
    }, index=dates)
    
    # Generate credit data
    credit_data = pd.DataFrame({
        'credit_spread': 0.01 + 0.005 * np.random.randn(len(dates))
    }, index=dates)
    
    market_data = {
        'prices': price_data,
        'options': options_data,
        'vix': vix_data,
        'rates': rates_data,
        'credit': credit_data
    }
    
    print(f"Generated sample data: {len(price_data)} days")
    return market_data


def run_strategy():
    """Run the complete options trading strategy."""
    print("=== OPTIONS TRADING STRATEGY ===")
    print("Starting strategy execution...")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy_config.yaml')
    config = StrategyConfig.from_yaml(config_path)
    print(f"Loaded configuration: {config.strategy_name} v{config.version}")
    
    # Load sample data
    market_data = load_sample_data()
    
    # Initialize components
    print("\nInitializing strategy components...")
    feature_builder = FeatureBuilder(config)
    forecaster = GBTForecaster(config)
    structure_selector = OptionStructureSelector(config)
    pricer = FDPricer(config)
    risk_manager = RiskManager(config)
    backtester = Backtester(config)
    
    # Build features
    print("\nBuilding features...")
    feature_set = feature_builder.build_features(
        price_data=market_data['prices'],
        options_data=market_data['options'],
        vix_data=market_data['vix'],
        rates_data=market_data['rates'],
        credit_data=market_data['credit']
    )
    
    print(f"Built {len(feature_set.feature_names)} features for {len(feature_set.features)} observations")
    
    # Prepare training data
    print("\nPreparing training data...")
    X = feature_set.features
    y = feature_set.target_direction  # Use direction as target
    
    # Remove rows with NaN targets
    mask = ~y.isna()
    X_train = X[mask]
    y_train = y[mask]
    
    print(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
    
    # Train model
    print("\nTraining forecasting model...")
    model_results = forecaster.train(X_train, y_train)
    
    print("Model training completed!")
    print(f"Training metrics: {model_results.metrics}")
    
    # Cross-validation
    print("\nRunning cross-validation...")
    cv_results = forecaster.cross_validate(X_train, y_train)
    
    print("Cross-validation completed!")
    print(f"CV metrics: {cv_results['average_metrics']}")
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = backtester.run_backtest(
        feature_data=feature_set.features,
        market_data=market_data,
        model=forecaster.model,
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2023-12-31')
    )
    
    # Generate reports
    print("\nGenerating reports...")
    generate_reports(backtest_results, forecaster, feature_builder)
    
    print("\nStrategy execution completed!")
    return backtest_results, forecaster, feature_builder


def generate_reports(backtest_results, forecaster, feature_builder):
    """Generate comprehensive strategy reports."""
    print("=== STRATEGY PERFORMANCE REPORT ===")
    
    # Performance metrics
    perf_metrics = backtest_results.performance_metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total Trades: {perf_metrics.get('total_trades', 0)}")
    print(f"  Hit Rate: {perf_metrics.get('hit_rate', 0):.2%}")
    print(f"  Total P&L: ${perf_metrics.get('total_pnl', 0):,.2f}")
    print(f"  Total Return: {perf_metrics.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Profit Factor: {perf_metrics.get('profit_factor', 0):.2f}")
    print(f"  Average Win: ${perf_metrics.get('avg_win', 0):.2f}")
    print(f"  Average Loss: ${perf_metrics.get('avg_loss', 0):.2f}")
    
    # Risk metrics
    risk_metrics = backtest_results.risk_metrics
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
    print(f"  VaR (95%): ${risk_metrics.get('var_95', 0):,.2f}")
    print(f"  VaR (99%): ${risk_metrics.get('var_99', 0):,.2f}")
    print(f"  Volatility: ${risk_metrics.get('volatility', 0):,.2f}")
    
    # Trade analysis
    trade_analysis = backtest_results.trade_analysis
    print(f"\nTrade Analysis:")
    print(f"  Total Trades: {trade_analysis.get('total_trades', 0)}")
    
    structure_analysis = trade_analysis.get('structure_analysis', {})
    if structure_analysis:
        print(f"  By Structure Type:")
        for structure_type, analysis in structure_analysis.items():
            print(f"    {structure_type}: {analysis['count']} trades, "
                  f"Hit Rate: {analysis['hit_rate']:.2%}, "
                  f"Avg P&L: ${analysis['avg_pnl']:.2f}")
    
    # Feature importance
    if hasattr(forecaster, 'model') and forecaster.model is not None:
        print(f"\nTop 10 Most Important Features:")
        try:
            importance_df = forecaster.get_feature_importance()
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            print(f"  Feature importance not available: {e}")
    
    # Create performance plots
    create_performance_plots(backtest_results)
    
    # Save results
    save_results(backtest_results, forecaster, feature_builder)


def create_performance_plots(backtest_results):
    """Create performance visualization plots."""
    print("Creating performance plots...")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Plot 1: Cumulative P&L
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if backtest_results.trades:
        trade_df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'cumulative_pnl': sum(t.pnl for t in backtest_results.trades[:i+1])
        } for i, t in enumerate(backtest_results.trades)])
        trade_df.set_index('timestamp', inplace=True)
        plt.plot(trade_df.index, trade_df['cumulative_pnl'])
        plt.title('Cumulative P&L')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Monthly returns
    plt.subplot(2, 2, 2)
    if not backtest_results.monthly_returns.empty:
        plt.bar(range(len(backtest_results.monthly_returns)), backtest_results.monthly_returns)
        plt.title('Monthly Returns')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    plt.subplot(2, 2, 3)
    if not backtest_results.drawdown_series.empty:
        plt.fill_between(backtest_results.drawdown_series.index, 
                        backtest_results.drawdown_series, 0, alpha=0.3, color='red')
        plt.title('Drawdown')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: P&L distribution
    plt.subplot(2, 2, 4)
    if backtest_results.trades:
        pnl_values = [t.pnl for t in backtest_results.trades]
        plt.hist(pnl_values, bins=20, alpha=0.7, edgecolor='black')
        plt.title('P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/strategy_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance plots saved to output/strategy_performance.png")


def save_results(backtest_results, forecaster, feature_builder):
    """Save strategy results to files."""
    print("Saving results...")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save trade summary
    if backtest_results.trades:
        trade_df = pd.DataFrame([{
            'trade_id': t.trade_id,
            'timestamp': t.timestamp,
            'structure_type': t.structure_type,
            'quantity': t.quantity,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'commission': t.commission,
            'slippage': t.slippage,
            'exit_reason': t.exit_reason
        } for t in backtest_results.trades])
        
        trade_df.to_csv('output/trade_summary.csv', index=False)
        print("Trade summary saved to output/trade_summary.csv")
    
    # Save performance metrics
    metrics_df = pd.DataFrame([
        {'metric': k, 'value': v} 
        for k, v in backtest_results.performance_metrics.items()
    ])
    metrics_df.to_csv('output/performance_metrics.csv', index=False)
    print("Performance metrics saved to output/performance_metrics.csv")
    
    # Save feature importance
    if hasattr(forecaster, 'model') and forecaster.model is not None:
        try:
            importance_df = forecaster.get_feature_importance()
            importance_df.to_csv('output/feature_importance.csv', index=False)
            print("Feature importance saved to output/feature_importance.csv")
        except Exception as e:
            print(f"Feature importance not available: {e}")
    
    print("Results saved successfully!")


def main():
    """Main execution function."""
    try:
        # Run the strategy
        backtest_results, forecaster, feature_builder = run_strategy()
        
        print("\n=== STRATEGY EXECUTION COMPLETED ===")
        print("Check the 'output/' directory for detailed results and plots.")
        
    except Exception as e:
        print(f"Error running strategy: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
