"""
Portfolio Risk Management

Aggregates portfolio Greeks and performs risk analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ..models.black_scholes import BlackScholesModel
from ..greeks.greeks_calculator import GreeksCalculator


class PortfolioRiskManager:
    """
    Manage and analyze risk for options portfolios.
    
    Capabilities:
    - Aggregate portfolio Greeks
    - Stress testing
    - Scenario analysis
    - VaR calculation
    """
    
    def __init__(self, r: float, current_time: float = 0.0):
        """
        Initialize portfolio risk manager.
        
        Args:
            r: Risk-free rate
            current_time: Current time (for time-to-maturity calculations)
        """
        self.r = r
        self.current_time = current_time
        self.positions = []
    
    def add_position(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
        quantity: int,
        position_type: str = 'long'
    ):
        """
        Add an option position to the portfolio.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            sigma: Volatility
            option_type: 'call' or 'put'
            quantity: Number of contracts
            position_type: 'long' or 'short'
        """
        position = {
            'S': S,
            'K': K,
            'T': T,
            'sigma': sigma,
            'option_type': option_type.lower(),
            'quantity': quantity,
            'position_type': position_type.lower()
        }
        self.positions.append(position)
    
    def calculate_portfolio_greeks(self, S: Optional[float] = None) -> Dict:
        """
        Calculate aggregated portfolio Greeks.
        
        Args:
            S: Stock price for calculation (uses position's S if None)
            
        Returns:
            Dictionary with portfolio Greeks
        """
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        total_rho = 0
        total_value = 0
        
        for pos in self.positions:
            # Use provided S or position's S
            stock_price = S if S is not None else pos['S']
            
            # Time to maturity
            time_to_maturity = max(pos['T'] - self.current_time, 1e-10)
            
            # Calculate Greeks
            greeks = GreeksCalculator(
                stock_price, pos['K'], time_to_maturity, self.r, pos['sigma']
            )
            
            # Calculate option value
            bs = BlackScholesModel(
                stock_price, pos['K'], time_to_maturity, self.r, pos['sigma']
            )
            option_value = bs.option_price(pos['option_type'])
            
            # Position multiplier
            multiplier = pos['quantity']
            if pos['position_type'] == 'short':
                multiplier *= -1
            
            # Aggregate Greeks
            total_delta += greeks.delta(pos['option_type']) * multiplier
            total_gamma += greeks.gamma() * multiplier
            total_vega += greeks.vega() * multiplier
            total_theta += greeks.theta(pos['option_type']) * multiplier
            total_rho += greeks.rho(pos['option_type']) * multiplier
            total_value += option_value * multiplier
        
        return {
            'portfolio_value': total_value,
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta,
            'rho': total_rho
        }
    
    def stress_test_volatility(
        self,
        vol_shocks: np.ndarray,
        S: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Stress test portfolio under volatility shocks.
        
        Args:
            vol_shocks: Array of volatility shock percentages (e.g., [-0.2, 0, 0.2])
            S: Stock price for calculation
            
        Returns:
            DataFrame with stress test results
        """
        results = []
        
        for shock in vol_shocks:
            # Create shocked positions
            shocked_positions = []
            for pos in self.positions:
                shocked_pos = pos.copy()
                shocked_pos['sigma'] = pos['sigma'] * (1 + shock)
                shocked_positions.append(shocked_pos)
            
            # Temporarily replace positions
            original_positions = self.positions
            self.positions = shocked_positions
            
            # Calculate Greeks under shock
            greeks = self.calculate_portfolio_greeks(S)
            
            # Restore original positions
            self.positions = original_positions
            
            results.append({
                'vol_shock': shock,
                'vol_shock_pct': shock * 100,
                'portfolio_value': greeks['portfolio_value'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'vega': greeks['vega']
            })
        
        return pd.DataFrame(results)
    
    def stress_test_price(
        self,
        price_shocks: np.ndarray
    ) -> pd.DataFrame:
        """
        Stress test portfolio under price shocks.
        
        Args:
            price_shocks: Array of price shock percentages
            
        Returns:
            DataFrame with stress test results
        """
        results = []
        
        # Get base stock price from first position
        if not self.positions:
            raise ValueError("No positions in portfolio")
        
        base_S = self.positions[0]['S']
        
        for shock in price_shocks:
            shocked_S = base_S * (1 + shock)
            greeks = self.calculate_portfolio_greeks(shocked_S)
            
            results.append({
                'price_shock': shock,
                'price_shock_pct': shock * 100,
                'stock_price': shocked_S,
                'portfolio_value': greeks['portfolio_value'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma']
            })
        
        return pd.DataFrame(results)
    
    def plot_greeks_profile(
        self,
        price_range: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot portfolio Greeks across stock prices.
        
        Args:
            price_range: Array of stock prices (auto-generated if None)
            save_path: Optional path to save figure
        """
        if not self.positions:
            raise ValueError("No positions in portfolio")
        
        # Generate price range if not provided
        if price_range is None:
            base_S = self.positions[0]['S']
            price_range = np.linspace(base_S * 0.7, base_S * 1.3, 100)
        
        # Calculate Greeks for each price
        deltas = []
        gammas = []
        vegas = []
        thetas = []
        
        for S in price_range:
            greeks = self.calculate_portfolio_greeks(S)
            deltas.append(greeks['delta'])
            gammas.append(greeks['gamma'])
            vegas.append(greeks['vega'])
            thetas.append(greeks['theta'])
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(price_range, deltas, 'b-', linewidth=2)
        axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Stock Price')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('Portfolio Delta')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(price_range, gammas, 'g-', linewidth=2)
        axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Stock Price')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Portfolio Gamma')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(price_range, vegas, 'purple', linewidth=2)
        axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Stock Price')
        axes[1, 0].set_ylabel('Vega')
        axes[1, 0].set_title('Portfolio Vega')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(price_range, thetas, 'orange', linewidth=2)
        axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Stock Price')
        axes[1, 1].set_ylabel('Theta')
        axes[1, 1].set_title('Portfolio Theta')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_risk_report(self) -> str:
        """
        Generate comprehensive risk report.
        
        Returns:
            Formatted risk report string
        """
        if not self.positions:
            return "No positions in portfolio"
        
        greeks = self.calculate_portfolio_greeks()
        
        report = "=" * 60 + "\n"
        report += "PORTFOLIO RISK REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Number of Positions: {len(self.positions)}\n"
        report += f"Portfolio Value: ${greeks['portfolio_value']:.2f}\n\n"
        
        report += "PORTFOLIO GREEKS:\n"
        report += "-" * 60 + "\n"
        report += f"Delta: {greeks['delta']:.4f}\n"
        report += f"Gamma: {greeks['gamma']:.4f}\n"
        report += f"Vega: {greeks['vega']:.4f}\n"
        report += f"Theta: {greeks['theta']:.4f} (per day)\n"
        report += f"Rho: {greeks['rho']:.4f}\n\n"
        
        report += "POSITION DETAILS:\n"
        report += "-" * 60 + "\n"
        for i, pos in enumerate(self.positions, 1):
            report += f"{i}. {pos['position_type'].upper()} {pos['quantity']} "
            report += f"{pos['option_type'].upper()} K={pos['K']:.2f} "
            report += f"T={pos['T']:.2f}y σ={pos['sigma']:.2%}\n"
        
        return report
