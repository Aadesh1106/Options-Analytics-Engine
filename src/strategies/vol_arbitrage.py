"""
Volatility Arbitrage Strategy

Identifies mispriced options and implements volatility arbitrage strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from ..models.black_scholes import BlackScholesModel
from ..volatility.implied_vol import ImpliedVolatilitySolver
from ..greeks.greeks_calculator import GreeksCalculator


class VolatilityArbitrageStrategy:
    """
    Volatility arbitrage strategy that trades mispriced options.
    
    Strategy:
    - Long options with implied vol < realized vol (undervalued)
    - Short options with implied vol > realized vol (overvalued)
    - Delta hedge to isolate volatility exposure
    """
    
    def __init__(
        self,
        S0: float,
        r: float,
        realized_vol: float,
        threshold: float = 0.05
    ):
        """
        Initialize volatility arbitrage strategy.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            realized_vol: Realized/forecast volatility
            threshold: Minimum vol difference to trade (e.g., 0.05 = 5%)
        """
        self.S0 = S0
        self.r = r
        self.realized_vol = realized_vol
        self.threshold = threshold
        self.positions = []
    
    def identify_mispricing(
        self,
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify mispriced options based on implied vs realized volatility.
        
        Args:
            options_data: DataFrame with columns ['K', 'T', 'market_price', 'option_type']
            
        Returns:
            DataFrame with mispricing analysis
        """
        results = []
        
        for idx, row in options_data.iterrows():
            K = row['K']
            T = row['T']
            market_price = row['market_price']
            option_type = row['option_type']
            
            # Solve for implied volatility
            solver = ImpliedVolatilitySolver(self.S0, K, T, self.r)
            try:
                implied_vol = solver.solve(market_price, option_type)
                
                # Calculate vol difference
                vol_diff = implied_vol - self.realized_vol
                vol_diff_pct = vol_diff / self.realized_vol
                
                # Determine if mispriced
                if abs(vol_diff_pct) > self.threshold:
                    if vol_diff > 0:
                        signal = 'SHORT'  # Overvalued
                    else:
                        signal = 'LONG'  # Undervalued
                else:
                    signal = 'NEUTRAL'
                
                results.append({
                    'K': K,
                    'T': T,
                    'option_type': option_type,
                    'market_price': market_price,
                    'implied_vol': implied_vol,
                    'realized_vol': self.realized_vol,
                    'vol_diff': vol_diff,
                    'vol_diff_pct': vol_diff_pct,
                    'signal': signal
                })
            except ValueError as e:
                # Could not solve for implied vol
                results.append({
                    'K': K,
                    'T': T,
                    'option_type': option_type,
                    'market_price': market_price,
                    'implied_vol': np.nan,
                    'realized_vol': self.realized_vol,
                    'vol_diff': np.nan,
                    'vol_diff_pct': np.nan,
                    'signal': 'ERROR'
                })
        
        return pd.DataFrame(results)
    
    def construct_portfolio(
        self,
        mispricing_df: pd.DataFrame,
        max_positions: int = 10
    ) -> List[Dict]:
        """
        Construct volatility arbitrage portfolio.
        
        Args:
            mispricing_df: DataFrame from identify_mispricing
            max_positions: Maximum number of positions
            
        Returns:
            List of position dictionaries
        """
        # Filter for tradeable signals
        tradeable = mispricing_df[mispricing_df['signal'].isin(['LONG', 'SHORT'])]
        
        # Sort by absolute vol difference
        tradeable = tradeable.sort_values('vol_diff_pct', key=abs, ascending=False)
        
        # Take top positions
        top_positions = tradeable.head(max_positions)
        
        portfolio = []
        for _, row in top_positions.iterrows():
            # Calculate delta for hedging
            greeks = GreeksCalculator(
                self.S0, row['K'], row['T'], self.r, row['implied_vol']
            )
            delta = greeks.delta(row['option_type'])
            
            position = {
                'K': row['K'],
                'T': row['T'],
                'option_type': row['option_type'],
                'signal': row['signal'],
                'option_price': row['market_price'],
                'implied_vol': row['implied_vol'],
                'vol_diff_pct': row['vol_diff_pct'],
                'delta': delta,
                'quantity': 1,  # Number of contracts
                'hedge_shares': -delta  # Shares to hedge
            }
            portfolio.append(position)
        
        return portfolio
    
    def backtest(
        self,
        portfolio: List[Dict],
        S_path: np.ndarray,
        time_points: np.ndarray,
        rebalance_frequency: int = 5
    ) -> Dict:
        """
        Backtest volatility arbitrage strategy.
        
        Args:
            portfolio: List of positions from construct_portfolio
            S_path: Simulated stock price path
            time_points: Time points for the path
            rebalance_frequency: Rebalance every N steps
            
        Returns:
            Dictionary with backtest results
        """
        n_steps = len(time_points)
        portfolio_value = np.zeros(n_steps)
        pnl = np.zeros(n_steps)
        
        # Initial portfolio value
        initial_value = 0
        for pos in portfolio:
            if pos['signal'] == 'LONG':
                initial_value -= pos['option_price'] * pos['quantity']
            else:  # SHORT
                initial_value += pos['option_price'] * pos['quantity']
            
            # Add hedge position
            initial_value -= pos['hedge_shares'] * self.S0
        
        portfolio_value[0] = initial_value
        
        # Simulate over time
        for i in range(1, n_steps):
            t = time_points[i]
            S = S_path[i]
            value = 0
            
            for pos in portfolio:
                time_to_maturity = max(pos['T'] - t, 1e-10)
                
                # Calculate current option value using realized vol
                bs = BlackScholesModel(
                    S, pos['K'], time_to_maturity, self.r, self.realized_vol
                )
                current_option_value = bs.option_price(pos['option_type'])
                
                # Option position value
                if pos['signal'] == 'LONG':
                    value += current_option_value * pos['quantity']
                else:  # SHORT
                    value -= current_option_value * pos['quantity']
                
                # Hedge position value
                value += pos['hedge_shares'] * S
                
                # Rebalance hedge if needed
                if i % rebalance_frequency == 0:
                    greeks = GreeksCalculator(
                        S, pos['K'], time_to_maturity, self.r, self.realized_vol
                    )
                    new_delta = greeks.delta(pos['option_type'])
                    pos['hedge_shares'] = -new_delta
            
            portfolio_value[i] = value
            pnl[i] = value - initial_value
        
        return {
            'time': time_points,
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'final_pnl': pnl[-1],
            'max_pnl': np.max(pnl),
            'min_pnl': np.min(pnl),
            'sharpe_ratio': np.mean(np.diff(pnl)) / np.std(np.diff(pnl)) if np.std(np.diff(pnl)) > 0 else 0
        }
    
    def plot_backtest_results(
        self,
        backtest_results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot backtest results.
        
        Args:
            backtest_results: Results from backtest method
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value
        axes[0].plot(
            backtest_results['time'],
            backtest_results['portfolio_value'],
            'b-',
            linewidth=2
        )
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Time (years)')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].set_title('Volatility Arbitrage Portfolio Value')
        axes[0].grid(True, alpha=0.3)
        
        # PnL
        axes[1].plot(
            backtest_results['time'],
            backtest_results['pnl'],
            'g-',
            linewidth=2
        )
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1].fill_between(
            backtest_results['time'],
            0,
            backtest_results['pnl'],
            alpha=0.3,
            color='g'
        )
        axes[1].set_xlabel('Time (years)')
        axes[1].set_ylabel('PnL')
        axes[1].set_title(
            f'Cumulative PnL (Final: {backtest_results["final_pnl"]:.2f}, '
            f'Sharpe: {backtest_results["sharpe_ratio"]:.2f})'
        )
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(
        self,
        backtest_results: Dict,
        portfolio: List[Dict]
    ) -> str:
        """
        Generate performance report.
        
        Args:
            backtest_results: Results from backtest
            portfolio: Portfolio positions
            
        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "VOLATILITY ARBITRAGE STRATEGY PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Number of Positions: {len(portfolio)}\n"
        report += f"Realized Volatility: {self.realized_vol:.2%}\n"
        report += f"Mispricing Threshold: {self.threshold:.2%}\n\n"
        
        report += "PERFORMANCE METRICS:\n"
        report += "-" * 60 + "\n"
        report += f"Final PnL: ${backtest_results['final_pnl']:.2f}\n"
        report += f"Maximum PnL: ${backtest_results['max_pnl']:.2f}\n"
        report += f"Minimum PnL: ${backtest_results['min_pnl']:.2f}\n"
        report += f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}\n\n"
        
        report += "POSITIONS:\n"
        report += "-" * 60 + "\n"
        for i, pos in enumerate(portfolio, 1):
            report += f"{i}. {pos['signal']} {pos['option_type'].upper()} "
            report += f"K={pos['K']:.2f} T={pos['T']:.2f}y "
            report += f"IV={pos['implied_vol']:.2%} "
            report += f"Diff={pos['vol_diff_pct']:.2%}\n"
        
        return report
