"""
Example 3: Implied Volatility and Volatility Surface

Demonstrates implied volatility solving and surface generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.black_scholes import BlackScholesModel
from src.volatility.implied_vol import ImpliedVolatilitySolver
from src.volatility.vol_surface import VolatilitySurface
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("IMPLIED VOLATILITY & VOLATILITY SURFACE")
    print("=" * 70)
    print()
    
    # Parameters
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    true_sigma = 0.25
    
    print("Step 1: Implied Volatility Solver")
    print("-" * 70)
    
    # Generate market price using true volatility
    bs = BlackScholesModel(S, K, T, r, true_sigma)
    market_price = bs.call_price()
    
    print(f"Market Call Price: ${market_price:.4f}")
    print(f"True Volatility: {true_sigma:.2%}")
    print()
    
    # Solve for implied volatility
    solver = ImpliedVolatilitySolver(S, K, T, r)
    implied_vol, diagnostics = solver.solve(
        market_price,
        'call',
        initial_guess=0.3,
        return_diagnostics=True
    )
    
    print(f"Implied Volatility: {implied_vol:.4%}")
    print(f"Converged: {diagnostics['converged']}")
    print(f"Iterations: {diagnostics['iterations']}")
    print(f"Final Error: ${diagnostics['final_error']:.8f}")
    print(f"Recovery Error: {abs(implied_vol - true_sigma):.8f}")
    print()
    
    # Show convergence
    print("Convergence History:")
    print(f"  {'Iter':<6} {'Sigma':<12} {'BS Price':<12} {'Error':<12}")
    print("  " + "-" * 42)
    for it in diagnostics['iteration_history'][:5]:  # Show first 5 iterations
        print(f"  {it['iteration']:<6} {it['sigma']:<12.6f} "
              f"{it['bs_price']:<12.6f} {it['price_diff']:<12.8f}")
    print()
    
    print("Step 2: Volatility Smile")
    print("-" * 70)
    
    # Generate volatility smile
    vol_surface = VolatilitySurface(S, r)
    strikes = np.linspace(80, 120, 20)
    implied_vols = vol_surface.generate_smile(
        T, strikes,
        base_vol=0.20,
        skew=0.10,
        convexity=0.05
    )
    
    # Plot smile
    vol_surface.plot_smile(T, strikes, implied_vols, 
                          save_path='../outputs/volatility_smile.png')
    print("Volatility smile plot saved to: outputs/volatility_smile.png")
    print()
    
    print("Step 3: 3D Volatility Surface")
    print("-" * 70)
    
    # Generate 3D surface
    strikes_3d = np.linspace(80, 120, 15)
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    
    surface = vol_surface.generate_surface(
        strikes_3d, maturities,
        base_vol=0.20,
        term_structure_slope=0.02
    )
    
    # Plot 3D surface
    vol_surface.plot_surface_3d(strikes_3d, maturities, surface,
                               save_path='../outputs/volatility_surface_3d.png')
    print("3D volatility surface saved to: outputs/volatility_surface_3d.png")
    
    # Plot heatmap
    vol_surface.plot_surface_heatmap(strikes_3d, maturities, surface,
                                    save_path='../outputs/volatility_surface_heatmap.png')
    print("Volatility surface heatmap saved to: outputs/volatility_surface_heatmap.png")
    print()
    
    # Display surface statistics
    print("Surface Statistics:")
    print(f"  Min Implied Vol: {np.min(surface):.2%}")
    print(f"  Max Implied Vol: {np.max(surface):.2%}")
    print(f"  Mean Implied Vol: {np.mean(surface):.2%}")
    print(f"  ATM Vol (T=1y): {surface[2, 7]:.2%}")  # Middle strike, middle maturity
    print()
    
    print("=" * 70)

if __name__ == "__main__":
    main()
