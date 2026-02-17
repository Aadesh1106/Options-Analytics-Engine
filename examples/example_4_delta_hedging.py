"""
Example 4: Delta Hedging Simulation

Demonstrates dynamic delta hedging of an option position.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.delta_hedging import DeltaHedgingSimulator
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("DELTA HEDGING SIMULATION")
    print("=" * 70)
    print()
    
    # Parameters
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    print("Parameters:")
    print(f"  Initial Stock Price: ${S0}")
    print(f"  Strike Price: ${K}")
    print(f"  Time to Maturity: {T} years")
    print(f"  Risk-Free Rate: {r:.1%}")
    print(f"  Volatility: {sigma:.1%}")
    print()
    
    print("Scenario: Short 1 Call Option, Delta Hedge Daily")
    print("-" * 70)
    
    # Run simulation
    simulator = DeltaHedgingSimulator(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type='call',
        position='short',
        rebalance_frequency='daily',
        random_seed=42
    )
    
    results = simulator.run_simulation()
    
    # Display results
    print(f"Initial Option Premium Received: ${simulator.initial_option_price:.4f}")
    print(f"Final Portfolio Value: ${results['portfolio_value'][-1]:.4f}")
    print(f"Final PnL: ${results['final_pnl']:.4f}")
    print(f"Max Hedging Error: ${results['max_hedging_error']:.4f}")
    print(f"Mean Hedging Error: ${results['mean_hedging_error']:.4f}")
    print()
    
    print(f"Final Stock Price: ${results['stock_price'][-1]:.2f}")
    print(f"Final Option Value: ${results['option_value'][-1]:.4f}")
    print(f"Final Delta: {results['delta'][-1]:.4f}")
    print(f"Final Hedge Position: {results['hedge_shares'][-1]:.4f} shares")
    print()
    
    # Plot results
    simulator.plot_results(save_path='../outputs/delta_hedging_simulation.png')
    print("Delta hedging plots saved to: outputs/delta_hedging_simulation.png")
    print()
    
    # Compare rebalancing frequencies
    print("Comparing Rebalancing Frequencies:")
    print("-" * 70)
    
    comparison = simulator.compare_frequencies(
        frequencies=['daily', 'weekly'],
        n_simulations=50
    )
    
    for freq, stats in comparison.items():
        print(f"\n{freq.upper()} Rebalancing:")
        print(f"  Mean PnL: ${stats['mean_pnl']:.4f}")
        print(f"  Std PnL: ${stats['std_pnl']:.4f}")
        print(f"  Mean Hedging Error: ${stats['mean_error']:.4f}")
        print(f"  Std Hedging Error: ${stats['std_error']:.4f}")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
