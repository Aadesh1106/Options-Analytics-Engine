"""
Example 6: Portfolio Risk Management

Demonstrates portfolio Greeks aggregation and stress testing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk.portfolio_risk import PortfolioRiskManager
import numpy as np

def main():
    print("=" * 70)
    print("PORTFOLIO RISK MANAGEMENT")
    print("=" * 70)
    print()
    
    # Initialize risk manager
    r = 0.05
    risk_manager = PortfolioRiskManager(r=r)
    
    print("Building Options Portfolio:")
    print("-" * 70)
    
    # Add positions
    positions = [
        {'S': 100, 'K': 95, 'T': 1.0, 'sigma': 0.20, 'type': 'call', 'qty': 10, 'pos': 'long'},
        {'S': 100, 'K': 100, 'T': 1.0, 'sigma': 0.22, 'type': 'call', 'qty': -5, 'pos': 'short'},
        {'S': 100, 'K': 105, 'T': 1.0, 'sigma': 0.24, 'type': 'call', 'qty': 10, 'pos': 'long'},
        {'S': 100, 'K': 100, 'T': 0.5, 'sigma': 0.21, 'type': 'put', 'qty': 5, 'pos': 'long'},
    ]
    
    for pos in positions:
        risk_manager.add_position(
            S=pos['S'],
            K=pos['K'],
            T=pos['T'],
            sigma=pos['sigma'],
            option_type=pos['type'],
            quantity=pos['qty'],
            position_type=pos['pos']
        )
        print(f"  {pos['pos'].upper()} {pos['qty']} {pos['type'].upper()} "
              f"K={pos['K']:.0f} T={pos['T']:.1f}y σ={pos['sigma']:.1%}")
    
    print()
    
    # Calculate portfolio Greeks
    print("Portfolio Greeks:")
    print("-" * 70)
    
    greeks = risk_manager.calculate_portfolio_greeks()
    
    print(f"  Portfolio Value: ${greeks['portfolio_value']:.2f}")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega: {greeks['vega']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f} (per day)")
    print(f"  Rho: {greeks['rho']:.4f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    if greeks['delta'] > 0:
        print(f"  • Portfolio is LONG {greeks['delta']:.2f} delta (benefits from price increase)")
    else:
        print(f"  • Portfolio is SHORT {abs(greeks['delta']):.2f} delta (benefits from price decrease)")
    
    if greeks['gamma'] > 0:
        print(f"  • Positive gamma: Delta increases as price rises")
    else:
        print(f"  • Negative gamma: Delta decreases as price rises")
    
    if greeks['vega'] > 0:
        print(f"  • Long vega: Benefits from volatility increase")
    else:
        print(f"  • Short vega: Benefits from volatility decrease")
    
    if greeks['theta'] < 0:
        print(f"  • Negative theta: Portfolio loses ${abs(greeks['theta']):.2f} per day from time decay")
    else:
        print(f"  • Positive theta: Portfolio gains from time decay")
    print()
    
    # Stress test - Price shocks
    print("Stress Test: Price Shocks")
    print("-" * 70)
    
    price_shocks = np.array([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
    price_stress = risk_manager.stress_test_price(price_shocks)
    
    print(price_stress.to_string(index=False))
    print()
    
    # Stress test - Volatility shocks
    print("Stress Test: Volatility Shocks")
    print("-" * 70)
    
    vol_shocks = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
    vol_stress = risk_manager.stress_test_volatility(vol_shocks)
    
    print(vol_stress.to_string(index=False))
    print()
    
    # Plot Greeks profile
    print("Generating Greeks Profile Plots...")
    risk_manager.plot_greeks_profile(
        save_path='../outputs/portfolio_greeks_profile.png'
    )
    print("Greeks profile saved to: outputs/portfolio_greeks_profile.png")
    print()
    
    # Generate risk report
    print("=" * 70)
    print(risk_manager.generate_risk_report())

if __name__ == "__main__":
    main()
