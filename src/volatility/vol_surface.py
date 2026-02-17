"""
Volatility Surface Generator

Generates implied volatility smiles and 3D volatility surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, Tuple
from .implied_vol import ImpliedVolatilitySolver
from ..models.black_scholes import BlackScholesModel


class VolatilitySurface:
    """
    Generate and visualize implied volatility surfaces.
    
    Creates volatility smiles across strikes and maturities.
    """
    
    def __init__(self, S: float, r: float, q: float = 0.0):
        """
        Initialize volatility surface generator.
        
        Args:
            S: Current stock price
            r: Risk-free rate
            q: Dividend yield
        """
        self.S = S
        self.r = r
        self.q = q
    
    def generate_smile(
        self,
        T: float,
        strikes: np.ndarray,
        base_vol: float = 0.2,
        skew: float = 0.1,
        convexity: float = 0.05
    ) -> np.ndarray:
        """
        Generate implied volatility smile.
        
        Uses a simple parametric model:
        σ(K) = base_vol + skew * (K/S - 1) + convexity * (K/S - 1)²
        
        Args:
            T: Time to maturity
            strikes: Array of strike prices
            base_vol: ATM volatility
            skew: Volatility skew parameter
            convexity: Smile convexity parameter
            
        Returns:
            Array of implied volatilities
        """
        moneyness = strikes / self.S
        implied_vols = base_vol + skew * (moneyness - 1) + convexity * (moneyness - 1)**2
        
        # Ensure volatilities are positive
        implied_vols = np.maximum(implied_vols, 0.01)
        
        return implied_vols
    
    def generate_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        base_vol: float = 0.2,
        term_structure_slope: float = 0.02
    ) -> np.ndarray:
        """
        Generate 3D volatility surface.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities (years)
            base_vol: Base ATM volatility
            term_structure_slope: Slope of term structure
            
        Returns:
            2D array of implied volatilities (maturities x strikes)
        """
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        surface = np.zeros((n_maturities, n_strikes))
        
        for i, T in enumerate(maturities):
            # Adjust base vol for term structure
            adjusted_base_vol = base_vol + term_structure_slope * T
            
            # Generate smile for this maturity
            surface[i, :] = self.generate_smile(
                T, strikes,
                base_vol=adjusted_base_vol,
                skew=0.1 * (1 - 0.3 * T),  # Skew flattens with maturity
                convexity=0.05 * (1 + 0.2 * T)  # Convexity increases with maturity
            )
        
        return surface
    
    def plot_smile(
        self,
        T: float,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot volatility smile.
        
        Args:
            T: Time to maturity
            strikes: Array of strikes
            implied_vols: Array of implied volatilities
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, implied_vols * 100, 'b-', linewidth=2)
        plt.axvline(self.S, color='r', linestyle='--', label='ATM', alpha=0.7)
        plt.xlabel('Strike Price', fontsize=12)
        plt.ylabel('Implied Volatility (%)', fontsize=12)
        plt.title(f'Volatility Smile (T={T:.2f} years)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_surface_3d(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        surface: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot 3D volatility surface.
        
        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            surface: 2D array of implied volatilities
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        K, T = np.meshgrid(strikes, maturities)
        
        surf = ax.plot_surface(K, T, surface * 100, cmap='viridis', alpha=0.9)
        
        ax.set_xlabel('Strike Price', fontsize=11)
        ax.set_ylabel('Time to Maturity (years)', fontsize=11)
        ax.set_zlabel('Implied Volatility (%)', fontsize=11)
        ax.set_title('Implied Volatility Surface', fontsize=14)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_surface_heatmap(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        surface: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot volatility surface as heatmap.
        
        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            surface: 2D array of implied volatilities
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            surface * 100,
            xticklabels=np.round(strikes, 1),
            yticklabels=np.round(maturities, 2),
            cmap='RdYlGn_r',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Implied Volatility (%)'}
        )
        plt.xlabel('Strike Price', fontsize=12)
        plt.ylabel('Time to Maturity (years)', fontsize=12)
        plt.title('Implied Volatility Surface Heatmap', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
