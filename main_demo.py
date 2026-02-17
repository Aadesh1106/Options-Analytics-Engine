"""
MAIN DEMO: Options Pricing & Volatility Trading Engine

Comprehensive demonstration of all features.
"""

import sys
sys.path.append('.')

from src.models.black_scholes import BlackScholesModel
from src.models.monte_carlo import MonteCarloEngine
from src.greeks.greeks_calculator import GreeksCalculator
from src.volatility.implied_vol import ImpliedVolatilitySolver
from src.volatility.vol_surface import VolatilitySurface
from src.strategies.delta_hedging import DeltaHedgingSimulator
from src.strategies.vol_arbitrage import VolatilityArbitrageStrategy
from src.simulation.path_simulator import PathSimulator
from src.risk.portfolio_risk import PortfolioRiskManager
import numpy as np
import pandas as pd

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def demo_black_scholes():
    """Demo Black-Scholes pricing."""
    print_section("1. BLACK-SCHOLES OPTION PRICING")
    
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    bs = BlackScholesModel(S, K, T, r, sigma)
    
    print(f"Parameters: S=${S}, K=${K}, T={T}y, r={r:.1%}, σ={sigma:.1%}")
    print(f"Call Price: ${bs.call_price():.4f}")
    print(f"Put Price: ${bs.put_price():.4f}")

def demo_greeks():
    """Demo Greeks calculation."""
    print_section("2. GREEKS CALCULATION")
    
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    greeks = GreeksCalculator(S, K, T, r, sigma)
    
    all_greeks = greeks.all_greeks('call')
    print("Call Option Greeks:")
    for name, value in all_greeks.items():
        print(f"  {name.capitalize()}: {value:.6f}")

def demo_monte_carlo():
    """Demo Monte Carlo pricing."""
    print_section("3. MONTE CARLO PRICING")
    
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    mc = MonteCarloEngine(S, K, T, r, sigma, n_simulations=100000, random_seed=42)
    bs = BlackScholesModel(S, K, T, r, sigma)
    
    mc_price, ci = mc.price_call()
    bs_price = bs.call_price()
    
    print(f"Monte Carlo Call Price: ${mc_price:.4f} ± ${ci:.4f}")
    print(f"Black-Scholes Price: ${bs_price:.4f}")
    print(f"Difference: ${abs(mc_price - bs_price):.4f}")
    print(f"Within Confidence Interval: {'✓' if abs(mc_price - bs_price) < ci else '✗'}")

def demo_implied_vol():
    """Demo implied volatility solver."""
    print_section("4. IMPLIED VOLATILITY SOLVER")
    
    S, K, T, r = 100, 100, 1.0, 0.05
    true_vol = 0.25
    
    # Generate market price
    bs = BlackScholesModel(S, K, T, r, true_vol)
    market_price = bs.call_price()
    
    # Solve for implied vol
    solver = ImpliedVolatilitySolver(S, K, T, r)
    implied_vol, diagnostics = solver.solve(market_price, 'call', return_diagnostics=True)
    
    print(f"Market Price: ${market_price:.4f}")
    print(f"True Volatility: {true_vol:.4%}")
    print(f"Implied Volatility: {implied_vol:.4%}")
    print(f"Iterations: {diagnostics['iterations']}")
    print(f"Recovery Error: {abs(implied_vol - true_vol):.8f}")

def demo_vol_surface():
    """Demo volatility surface."""
    print_section("5. VOLATILITY SURFACE")
    
    S, r = 100, 0.05
    vol_surface = VolatilitySurface(S, r)
    
    strikes = np.linspace(80, 120, 10)
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    
    surface = vol_surface.generate_surface(strikes, maturities)
    
    print(f"Surface Shape: {surface.shape} (maturities × strikes)")
    print(f"Min Vol: {np.min(surface):.2%}")
    print(f"Max Vol: {np.max(surface):.2%}")
    print(f"ATM Vol (T=1y): {surface[2, 5]:.2%}")

def demo_delta_hedging():
    """Demo delta hedging."""
    print_section("6. DELTA HEDGING SIMULATION")
    
    simulator = DeltaHedgingSimulator(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.20,
        option_type='call', position='short',
        rebalance_frequency='daily', random_seed=42
    )
    
    results = simulator.run_simulation()
    
    print(f"Initial Premium: ${simulator.initial_option_price:.4f}")
    print(f"Final PnL: ${results['final_pnl']:.4f}")
    print(f"Max Hedging Error: ${results['max_hedging_error']:.4f}")
    print(f"Mean Hedging Error: ${results['mean_hedging_error']:.4f}")

def demo_vol_arbitrage():
    """Demo volatility arbitrage."""
    print_section("7. VOLATILITY ARBITRAGE")
    
    S0, r, realized_vol = 100, 0.05, 0.20
    
    # Create synthetic market
    options_data = pd.DataFrame({
        'K': [95, 100, 105],
        'T': [1.0, 1.0, 1.0],
        'market_price': [8.5, 5.5, 3.2],
        'option_type': ['call', 'call', 'call']
    })
    
    strategy = VolatilityArbitrageStrategy(S0, r, realized_vol, threshold=0.05)
    mispricing = strategy.identify_mispricing(options_data)
    
    print("Mispricing Analysis:")
    print(mispricing[['K', 'implied_vol', 'vol_diff_pct', 'signal']].to_string(index=False))

def demo_portfolio_risk():
    """Demo portfolio risk management."""
    print_section("8. PORTFOLIO RISK MANAGEMENT")
    
    risk_manager = PortfolioRiskManager(r=0.05)
    
    # Add positions
    risk_manager.add_position(100, 95, 1.0, 0.20, 'call', 10, 'long')
    risk_manager.add_position(100, 100, 1.0, 0.22, 'call', -5, 'short')
    risk_manager.add_position(100, 105, 1.0, 0.24, 'put', 5, 'long')
    
    greeks = risk_manager.calculate_portfolio_greeks()
    
    print(f"Portfolio Value: ${greeks['portfolio_value']:.2f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f} (per day)")
    
    # Stress test
    price_shocks = np.array([-0.10, 0.0, 0.10])
    stress = risk_manager.stress_test_price(price_shocks)
    print("\nPrice Stress Test:")
    print(stress[['price_shock_pct', 'portfolio_value']].to_string(index=False))

def main():
    """Run all demos."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  OPTIONS PRICING & VOLATILITY TRADING ENGINE - COMPREHENSIVE DEMO".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        demo_black_scholes()
        demo_greeks()
        demo_monte_carlo()
        demo_implied_vol()
        demo_vol_surface()
        demo_delta_hedging()
        demo_vol_arbitrage()
        demo_portfolio_risk()
        
        print_section("DEMO COMPLETE")
        print("✓ All features demonstrated successfully!")
        print("\nFor detailed examples, run scripts in the examples/ directory:")
        print("  • example_1_black_scholes.py")
        print("  • example_2_monte_carlo.py")
        print("  • example_3_implied_vol.py")
        print("  • example_4_delta_hedging.py")
        print("  • example_5_vol_arbitrage.py")
        print("  • example_6_portfolio_risk.py")
        print("\n" + "█" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
