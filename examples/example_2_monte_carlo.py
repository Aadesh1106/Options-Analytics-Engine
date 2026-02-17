"""
Example 2: Monte Carlo Pricing vs Black-Scholes

Compares Monte Carlo pricing with analytical Black-Scholes pricing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.black_scholes import BlackScholesModel
from src.models.monte_carlo import MonteCarloEngine
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 70)
    print("MONTE CARLO PRICING VS BLACK-SCHOLES")
    print("=" * 70)
    print()
    
    # Parameters
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    n_simulations = 100000
    
    print("Parameters:")
    print(f"  Stock Price: ${S}")
    print(f"  Strike Price: ${K}")
    print(f"  Time to Maturity: {T} years")
    print(f"  Risk-Free Rate: {r:.1%}")
    print(f"  Volatility: {sigma:.1%}")
    print(f"  Monte Carlo Simulations: {n_simulations:,}")
    print()
    
    # Black-Scholes pricing
    print("Black-Scholes Pricing:")
    bs = BlackScholesModel(S, K, T, r, sigma)
    bs_call = bs.call_price()
    bs_put = bs.put_price()
    print(f"  Call Price: ${bs_call:.4f}")
    print(f"  Put Price: ${bs_put:.4f}")
    print()
    
    # Monte Carlo pricing
    print("Monte Carlo Pricing:")
    mc = MonteCarloEngine(S, K, T, r, sigma, n_simulations, random_seed=42)
    
    # Price call
    mc_call, mc_call_ci = mc.price_call(use_antithetic=True)
    print(f"  Call Price: ${mc_call:.4f} ± ${mc_call_ci:.4f}")
    print(f"  BS Call Price: ${bs_call:.4f}")
    print(f"  Difference: ${abs(mc_call - bs_call):.4f}")
    print(f"  Within CI: {'✓' if abs(mc_call - bs_call) < mc_call_ci else '✗'}")
    print()
    
    # Price put
    mc_put, mc_put_ci = mc.price_put(use_antithetic=True)
    print(f"  Put Price: ${mc_put:.4f} ± ${mc_put_ci:.4f}")
    print(f"  BS Put Price: ${bs_put:.4f}")
    print(f"  Difference: ${abs(mc_put - bs_put):.4f}")
    print(f"  Within CI: {'✓' if abs(mc_put - bs_put) < mc_put_ci else '✗'}")
    print()
    
    # Convergence analysis
    print("Running convergence analysis...")
    n_sims, prices = mc.convergence_analysis('call', n_steps=10)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogx(n_sims, prices, 'b-o', label='Monte Carlo', linewidth=2, markersize=8)
    plt.axhline(bs_call, color='r', linestyle='--', label='Black-Scholes', linewidth=2)
    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('Call Option Price', fontsize=12)
    plt.title('Monte Carlo Convergence to Black-Scholes Price', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../outputs/monte_carlo_convergence.png', dpi=300, bbox_inches='tight')
    print("Convergence plot saved to: outputs/monte_carlo_convergence.png")
    plt.show()
    
    # Comparison with/without antithetic variates
    print("\nComparing Variance Reduction Techniques:")
    mc_no_anti, ci_no_anti = mc.price_call(use_antithetic=False)
    mc_anti, ci_anti = mc.price_call(use_antithetic=True)
    
    print(f"  Without Antithetic Variates: ${mc_no_anti:.4f} ± ${ci_no_anti:.4f}")
    print(f"  With Antithetic Variates: ${mc_anti:.4f} ± ${ci_anti:.4f}")
    print(f"  CI Reduction: {(1 - ci_anti/ci_no_anti)*100:.1f}%")
    print()
    
    print("=" * 70)

if __name__ == "__main__":
    main()
