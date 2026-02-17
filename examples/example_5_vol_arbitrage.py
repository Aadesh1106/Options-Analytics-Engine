"""
Example 5: Volatility Arbitrage Strategy

Demonstrates volatility arbitrage trading strategy.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.vol_arbitrage import VolatilityArbitrageStrategy
from src.simulation.path_simulator import PathSimulator
import pandas as pd
import numpy as np

def main():
    print("=" * 70)
    print("VOLATILITY ARBITRAGE STRATEGY")
    print("=" * 70)
    print()
    
    # Market parameters
    S0 = 100
    r = 0.05
    realized_vol = 0.20  # Our forecast: 20% volatility
    
    print("Market Setup:")
    print(f"  Stock Price: ${S0}")
    print(f"  Risk-Free Rate: {r:.1%}")
    print(f"  Realized/Forecast Volatility: {realized_vol:.1%}")
    print()
    
    # Create synthetic options market data
    print("Step 1: Market Options Data")
    print("-" * 70)
    
    options_data = pd.DataFrame({
        'K': [90, 95, 100, 105, 110, 95, 100, 105],
        'T': [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        'market_price': [12.5, 8.2, 5.0, 2.8, 1.3, 10.5, 7.5, 5.2],
        'option_type': ['call', 'call', 'call', 'call', 'call', 'call', 'call', 'call']
    })
    
    print(options_data.to_string(index=False))
    print()
    
    # Initialize strategy
    strategy = VolatilityArbitrageStrategy(
        S0=S0,
        r=r,
        realized_vol=realized_vol,
        threshold=0.05  # Trade if vol difference > 5%
    )
    
    # Identify mispricing
    print("Step 2: Identify Mispricing")
    print("-" * 70)
    
    mispricing = strategy.identify_mispricing(options_data)
    
    print("\nMispricing Analysis:")
    print(mispricing[['K', 'T', 'implied_vol', 'vol_diff_pct', 'signal']].to_string(index=False))
    print()
    
    # Count signals
    signals = mispricing['signal'].value_counts()
    print("Signal Distribution:")
    for signal, count in signals.items():
        print(f"  {signal}: {count}")
    print()
    
    # Construct portfolio
    print("Step 3: Construct Portfolio")
    print("-" * 70)
    
    portfolio = strategy.construct_portfolio(mispricing, max_positions=5)
    
    print(f"\nPortfolio: {len(portfolio)} positions")
    for i, pos in enumerate(portfolio, 1):
        print(f"\n{i}. {pos['signal']} {pos['option_type'].upper()}")
        print(f"   Strike: ${pos['K']:.0f}, Maturity: {pos['T']:.1f}y")
        print(f"   Implied Vol: {pos['implied_vol']:.2%}")
        print(f"   Vol Difference: {pos['vol_diff_pct']:.2%}")
        print(f"   Delta: {pos['delta']:.4f}")
        print(f"   Hedge: {pos['hedge_shares']:.4f} shares")
    print()
    
    # Backtest
    print("Step 4: Backtest Strategy")
    print("-" * 70)
    
    # Simulate stock price path
    simulator = PathSimulator(
        S0=S0,
        mu=r,  # Risk-neutral drift
        sigma=realized_vol,
        T=1.0,
        n_steps=252,
        random_seed=42
    )
    
    time_points, S_path = simulator.simulate_path()
    
    # Run backtest
    backtest_results = strategy.backtest(
        portfolio=portfolio,
        S_path=S_path,
        time_points=time_points,
        rebalance_frequency=5  # Rebalance every 5 days
    )
    
    # Display results
    print("\nBacktest Results:")
    print(f"  Final PnL: ${backtest_results['final_pnl']:.2f}")
    print(f"  Maximum PnL: ${backtest_results['max_pnl']:.2f}")
    print(f"  Minimum PnL: ${backtest_results['min_pnl']:.2f}")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print()
    
    # Plot results
    strategy.plot_backtest_results(
        backtest_results,
        save_path='../outputs/vol_arbitrage_backtest.png'
    )
    print("Backtest plots saved to: outputs/vol_arbitrage_backtest.png")
    print()
    
    # Generate report
    print("Step 5: Performance Report")
    print("-" * 70)
    
    report = strategy.generate_performance_report(backtest_results, portfolio)
    print(report)
    
    print("=" * 70)

if __name__ == "__main__":
    main()
