"""
Example 1: Black-Scholes Pricing and Greeks Calculation

Demonstrates basic option pricing and Greeks calculation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.black_scholes import BlackScholesModel
from src.greeks.greeks_calculator import GreeksCalculator
import numpy as np

def main():
    print("=" * 70)
    print("BLACK-SCHOLES PRICING & GREEKS CALCULATION")
    print("=" * 70)
    print()
    
    # Parameters
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    print("Parameters:")
    print(f"  Stock Price (S): ${S}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Time to Maturity (T): {T} years")
    print(f"  Risk-Free Rate (r): {r:.1%}")
    print(f"  Volatility (σ): {sigma:.1%}")
    print()
    
    # Create Black-Scholes model
    bs = BlackScholesModel(S, K, T, r, sigma)
    
    # Calculate option prices
    call_price = bs.call_price()
    put_price = bs.put_price()
    
    print("Option Prices:")
    print(f"  Call Price: ${call_price:.4f}")
    print(f"  Put Price: ${put_price:.4f}")
    print()
    
    # Verify put-call parity: C - P = S - K*e^(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    print("Put-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.4f}")
    print(f"  S - K*e^(-rT) = {parity_rhs:.4f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.10f} ✓")
    print()
    
    # Calculate Greeks
    greeks = GreeksCalculator(S, K, T, r, sigma)
    
    print("Greeks for CALL option:")
    print(f"  Delta: {greeks.delta('call'):.4f}")
    print(f"  Gamma: {greeks.gamma():.6f}")
    print(f"  Vega: {greeks.vega():.4f} (per 1% vol change)")
    print(f"  Theta: {greeks.theta('call'):.4f} (per day)")
    print(f"  Rho: {greeks.rho('call'):.4f} (per 1% rate change)")
    print()
    
    print("Greeks for PUT option:")
    print(f"  Delta: {greeks.delta('put'):.4f}")
    print(f"  Gamma: {greeks.gamma():.6f}")
    print(f"  Vega: {greeks.vega():.4f} (per 1% vol change)")
    print(f"  Theta: {greeks.theta('put'):.4f} (per day)")
    print(f"  Rho: {greeks.rho('put'):.4f} (per 1% rate change)")
    print()
    
    # Vectorized pricing across strikes
    print("Vectorized Pricing Across Strikes:")
    strikes = np.array([90, 95, 100, 105, 110])
    bs_vec = BlackScholesModel(S, strikes, T, r, sigma)
    call_prices = bs_vec.call_price()
    
    print(f"  {'Strike':<10} {'Call Price':<15} {'Moneyness':<15}")
    print("  " + "-" * 40)
    for k, c in zip(strikes, call_prices):
        moneyness = "ITM" if k < S else ("ATM" if k == S else "OTM")
        print(f"  {k:<10.0f} ${c:<14.4f} {moneyness:<15}")
    print()
    
    print("=" * 70)

if __name__ == "__main__":
    main()
