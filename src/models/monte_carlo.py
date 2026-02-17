"""
Monte Carlo Option Pricing Engine

Simulates risk-neutral price paths to price options using Monte Carlo methods.
Includes variance reduction techniques and convergence analysis.
"""

import numpy as np
from typing import Tuple, Optional
from ..utils.helpers import validate_option_params, confidence_interval


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for option pricing.
    
    Simulates stock price paths under risk-neutral measure:
    S(T) = S₀ * exp[(r - q - σ²/2)T + σ√T * Z]
    where Z ~ N(0,1)
    
    Attributes:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        n_simulations: Number of Monte Carlo paths
        q: Dividend yield (default 0)
        random_seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_simulations: int = 100000,
        q: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            n_simulations: Number of simulation paths (default 100,000)
            q: Dividend yield (default 0)
            random_seed: Random seed for reproducibility
        """
        validate_option_params(S, K, T, r, sigma)
        
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.q = q
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_terminal_prices(self) -> np.ndarray:
        """
        Simulate terminal stock prices at maturity.
        
        Uses geometric Brownian motion under risk-neutral measure:
        S(T) = S₀ * exp[(r - q - σ²/2)T + σ√T * Z]
        
        Returns:
            Array of terminal stock prices
        """
        # Generate random normal variables
        Z = np.random.standard_normal(self.n_simulations)
        
        # Calculate terminal prices
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        
        S_T = self.S * np.exp(drift + diffusion)
        
        return S_T
    
    def simulate_terminal_prices_antithetic(self) -> np.ndarray:
        """
        Simulate terminal prices using antithetic variates for variance reduction.
        
        For each random draw Z, also use -Z to reduce variance.
        
        Returns:
            Array of terminal stock prices
        """
        # Generate half the required random variables
        n_half = self.n_simulations // 2
        Z = np.random.standard_normal(n_half)
        
        # Create antithetic pairs
        Z_antithetic = np.concatenate([Z, -Z])
        
        # Calculate terminal prices
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z_antithetic
        
        S_T = self.S * np.exp(drift + diffusion)
        
        return S_T
    
    def price_call(
        self,
        use_antithetic: bool = True,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Price European call option using Monte Carlo.
        
        Args:
            use_antithetic: Use antithetic variates for variance reduction
            confidence_level: Confidence level for interval (default 0.95)
            
        Returns:
            Tuple of (option_price, confidence_interval_half_width)
        """
        # Simulate terminal prices
        if use_antithetic:
            S_T = self.simulate_terminal_prices_antithetic()
        else:
            S_T = self.simulate_terminal_prices()
        
        # Calculate payoffs
        payoffs = np.maximum(S_T - self.K, 0)
        
        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs
        
        # Calculate price and confidence interval
        price = np.mean(discounted_payoffs)
        ci_lower, ci_upper = confidence_interval(discounted_payoffs, confidence_level)
        ci_half_width = (ci_upper - ci_lower) / 2
        
        return price, ci_half_width
    
    def price_put(
        self,
        use_antithetic: bool = True,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Price European put option using Monte Carlo.
        
        Args:
            use_antithetic: Use antithetic variates for variance reduction
            confidence_level: Confidence level for interval (default 0.95)
            
        Returns:
            Tuple of (option_price, confidence_interval_half_width)
        """
        # Simulate terminal prices
        if use_antithetic:
            S_T = self.simulate_terminal_prices_antithetic()
        else:
            S_T = self.simulate_terminal_prices()
        
        # Calculate payoffs
        payoffs = np.maximum(self.K - S_T, 0)
        
        # Discount to present value
        discount_factor = np.exp(-self.r * self.T)
        discounted_payoffs = discount_factor * payoffs
        
        # Calculate price and confidence interval
        price = np.mean(discounted_payoffs)
        ci_lower, ci_upper = confidence_interval(discounted_payoffs, confidence_level)
        ci_half_width = (ci_upper - ci_lower) / 2
        
        return price, ci_half_width
    
    def convergence_analysis(
        self,
        option_type: str,
        n_steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze convergence of Monte Carlo pricing.
        
        Args:
            option_type: 'call' or 'put'
            n_steps: Number of convergence steps
            
        Returns:
            Tuple of (n_simulations_array, prices_array)
        """
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        # Create logarithmically spaced simulation counts
        n_sims = np.logspace(2, np.log10(self.n_simulations), n_steps, dtype=int)
        prices = np.zeros(n_steps)
        
        # Store original n_simulations
        original_n_sims = self.n_simulations
        
        for i, n in enumerate(n_sims):
            self.n_simulations = n
            
            if option_type == 'call':
                price, _ = self.price_call(use_antithetic=False)
            else:
                price, _ = self.price_put(use_antithetic=False)
            
            prices[i] = price
        
        # Restore original n_simulations
        self.n_simulations = original_n_sims
        
        return n_sims, prices
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MonteCarloEngine(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, n_simulations={self.n_simulations})"
        )
