"""
Stock Price Path Simulator

Simulates stock price paths using geometric Brownian motion.
"""

import numpy as np
from typing import Optional, Tuple


class PathSimulator:
    """
    Simulate stock price paths using geometric Brownian motion.
    
    dS = μS dt + σS dW
    
    where W is a Wiener process (Brownian motion).
    """
    
    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int,
        random_seed: Optional[int] = None
    ):
        """
        Initialize path simulator.
        
        Args:
            S0: Initial stock price
            mu: Drift (expected return)
            sigma: Volatility
            T: Time horizon (years)
            n_steps: Number of time steps
            random_seed: Random seed for reproducibility
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a single price path.
        
        Returns:
            Tuple of (time_points, price_path)
        """
        # Time points
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(self.dt), self.n_steps)
        
        # Initialize price path
        S = np.zeros(self.n_steps + 1)
        S[0] = self.S0
        
        # Simulate path
        for i in range(self.n_steps):
            S[i + 1] = S[i] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt +
                self.sigma * dW[i]
            )
        
        return t, S
    
    def simulate_paths(self, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple price paths.
        
        Args:
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (time_points, price_paths)
            price_paths shape: (n_paths, n_steps + 1)
        """
        # Time points
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        # Initialize price paths
        S = np.zeros((n_paths, self.n_steps + 1))
        S[:, 0] = self.S0
        
        # Generate all random shocks at once
        dW = np.random.normal(0, np.sqrt(self.dt), (n_paths, self.n_steps))
        
        # Simulate all paths
        for i in range(self.n_steps):
            S[:, i + 1] = S[:, i] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt +
                self.sigma * dW[:, i]
            )
        
        return t, S
    
    def simulate_path_with_returns(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate path and calculate returns.
        
        Returns:
            Tuple of (time_points, price_path, returns)
        """
        t, S = self.simulate_path()
        returns = np.diff(np.log(S))
        return t, S, returns
