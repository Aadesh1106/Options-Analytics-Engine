"""
Delta Hedging Simulator

Simulates dynamic delta hedging strategy for options.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from ..simulation.path_simulator import PathSimulator
from ..models.black_scholes import BlackScholesModel
from ..greeks.greeks_calculator import GreeksCalculator


class DeltaHedgingSimulator:
    """
    Simulate dynamic delta hedging of an option position.
    
    Delta hedging involves holding a position in the underlying asset
    to offset changes in the option value due to price movements.
    
    For a short call position, we buy delta shares of the underlying.
    The portfolio value is: V = -C + Δ*S + B
    where B is cash in risk-free account.
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        position: str = 'short',
        rebalance_frequency: str = 'daily',
        n_steps: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize delta hedging simulator.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            position: 'long' or 'short' option position
            rebalance_frequency: 'daily', 'weekly', or 'continuous'
            n_steps: Number of simulation steps (auto-calculated if None)
            random_seed: Random seed
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.position = position.lower()
        self.random_seed = random_seed
        
        # Determine number of steps based on rebalancing frequency
        if n_steps is None:
            if rebalance_frequency == 'daily':
                self.n_steps = int(T * 252)  # 252 trading days per year
            elif rebalance_frequency == 'weekly':
                self.n_steps = int(T * 52)  # 52 weeks per year
            elif rebalance_frequency == 'continuous':
                self.n_steps = int(T * 252 * 7)  # Hourly rebalancing
            else:
                raise ValueError(f"Unknown rebalance_frequency: {rebalance_frequency}")
        else:
            self.n_steps = n_steps
        
        self.dt = T / self.n_steps
        
        # Calculate initial option price
        bs = BlackScholesModel(S0, K, T, r, sigma)
        self.initial_option_price = bs.option_price(option_type)
        
        # Results storage
        self.results = None
    
    def run_simulation(self) -> Dict:
        """
        Run delta hedging simulation.
        
        Returns:
            Dictionary with simulation results
        """
        # Simulate stock price path
        simulator = PathSimulator(
            self.S0, self.r, self.sigma, self.T, self.n_steps, self.random_seed
        )
        t, S = simulator.simulate_path()
        
        # Initialize arrays
        n = len(t)
        delta = np.zeros(n)
        option_value = np.zeros(n)
        hedge_shares = np.zeros(n)
        cash_account = np.zeros(n)
        portfolio_value = np.zeros(n)
        hedging_error = np.zeros(n)
        pnl = np.zeros(n)
        
        # Position multiplier (-1 for short, +1 for long)
        pos_mult = -1 if self.position == 'short' else 1
        
        # Initial setup
        time_to_maturity = self.T - t[0]
        bs = BlackScholesModel(S[0], self.K, time_to_maturity, self.r, self.sigma)
        greeks = GreeksCalculator(S[0], self.K, time_to_maturity, self.r, self.sigma)
        
        option_value[0] = bs.option_price(self.option_type)
        delta[0] = greeks.delta(self.option_type)
        hedge_shares[0] = -pos_mult * delta[0]  # Opposite sign to hedge
        cash_account[0] = pos_mult * option_value[0] - hedge_shares[0] * S[0]
        portfolio_value[0] = pos_mult * option_value[0]
        
        # Simulate hedging over time
        for i in range(1, n):
            time_to_maturity = max(self.T - t[i], 1e-10)  # Avoid division by zero
            
            # Calculate option value and delta
            bs = BlackScholesModel(S[i], self.K, time_to_maturity, self.r, self.sigma)
            greeks = GreeksCalculator(S[i], self.K, time_to_maturity, self.r, self.sigma)
            
            option_value[i] = bs.option_price(self.option_type)
            delta[i] = greeks.delta(self.option_type)
            
            # Update cash account (accrue interest)
            cash_account[i] = cash_account[i-1] * np.exp(self.r * self.dt)
            
            # Rebalance hedge
            new_hedge_shares = -pos_mult * delta[i]
            shares_to_buy = new_hedge_shares - hedge_shares[i-1]
            cash_account[i] -= shares_to_buy * S[i]
            hedge_shares[i] = new_hedge_shares
            
            # Calculate portfolio value
            portfolio_value[i] = (
                pos_mult * option_value[i] +
                hedge_shares[i] * S[i] +
                cash_account[i]
            )
            
            # Hedging error (should be close to initial option premium)
            hedging_error[i] = portfolio_value[i] - portfolio_value[0]
            
            # PnL
            pnl[i] = portfolio_value[i] - portfolio_value[0]
        
        # Store results
        self.results = {
            'time': t,
            'stock_price': S,
            'option_value': option_value,
            'delta': delta,
            'hedge_shares': hedge_shares,
            'cash_account': cash_account,
            'portfolio_value': portfolio_value,
            'hedging_error': hedging_error,
            'pnl': pnl,
            'final_pnl': pnl[-1],
            'max_hedging_error': np.max(np.abs(hedging_error)),
            'mean_hedging_error': np.mean(np.abs(hedging_error))
        }
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot hedging simulation results.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.results is None:
            raise ValueError("Must run simulation before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Stock price path
        axes[0, 0].plot(self.results['time'], self.results['stock_price'], 'b-', linewidth=1.5)
        axes[0, 0].axhline(self.K, color='r', linestyle='--', label='Strike', alpha=0.7)
        axes[0, 0].set_xlabel('Time (years)')
        axes[0, 0].set_ylabel('Stock Price')
        axes[0, 0].set_title('Stock Price Path')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Delta evolution
        axes[0, 1].plot(self.results['time'], self.results['delta'], 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (years)')
        axes[0, 1].set_ylabel('Delta')
        axes[0, 1].set_title('Option Delta Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Portfolio value
        axes[1, 0].plot(self.results['time'], self.results['portfolio_value'], 'purple', linewidth=1.5)
        axes[1, 0].axhline(self.results['portfolio_value'][0], color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Time (years)')
        axes[1, 0].set_ylabel('Portfolio Value')
        axes[1, 0].set_title('Hedged Portfolio Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # PnL
        axes[1, 1].plot(self.results['time'], self.results['pnl'], 'orange', linewidth=1.5)
        axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Time (years)')
        axes[1, 1].set_ylabel('PnL')
        axes[1, 1].set_title(f'Hedging PnL (Final: {self.results["final_pnl"]:.4f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_frequencies(
        self,
        frequencies: list = ['daily', 'weekly'],
        n_simulations: int = 100
    ) -> Dict:
        """
        Compare hedging performance across different rebalancing frequencies.
        
        Args:
            frequencies: List of rebalancing frequencies
            n_simulations: Number of simulations per frequency
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for freq in frequencies:
            pnls = []
            errors = []
            
            for _ in range(n_simulations):
                sim = DeltaHedgingSimulator(
                    self.S0, self.K, self.T, self.r, self.sigma,
                    self.option_type, self.position, freq
                )
                res = sim.run_simulation()
                pnls.append(res['final_pnl'])
                errors.append(res['max_hedging_error'])
            
            results[freq] = {
                'mean_pnl': np.mean(pnls),
                'std_pnl': np.std(pnls),
                'mean_error': np.mean(errors),
                'std_error': np.std(errors)
            }
        
        return results
