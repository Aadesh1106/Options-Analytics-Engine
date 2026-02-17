"""
Implied Volatility Solver

Solves for implied volatility using Newton-Raphson method with numerical stability.
"""

import numpy as np
from typing import Optional, Tuple, Union
from ..models.black_scholes import BlackScholesModel
from ..greeks.greeks_calculator import GreeksCalculator


class ImpliedVolatilitySolver:
    """
    Solve for implied volatility using Newton-Raphson method.
    
    Given a market price, finds the volatility that makes the Black-Scholes
    price equal to the market price.
    
    Uses Vega (∂V/∂σ) for Newton-Raphson iterations:
    σ_new = σ_old - (V_BS(σ_old) - V_market) / Vega(σ_old)
    
    Attributes:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0
    ):
        """
        Initialize implied volatility solver.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            q: Dividend yield (default 0)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
    
    def solve(
        self,
        market_price: float,
        option_type: str,
        initial_guess: float = 0.3,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        return_diagnostics: bool = False
    ) -> Union[float, Tuple[float, dict]]:
        """
        Solve for implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price of option
            option_type: 'call' or 'put'
            initial_guess: Initial volatility guess (default 0.3 = 30%)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            return_diagnostics: Return convergence diagnostics
            
        Returns:
            Implied volatility, or (implied_vol, diagnostics) if return_diagnostics=True
            
        Raises:
            ValueError: If convergence fails or inputs are invalid
        """
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        # Check if option has intrinsic value
        intrinsic = max(self.S - self.K, 0) if option_type == 'call' else max(self.K - self.S, 0)
        if market_price < intrinsic:
            raise ValueError(
                f"Market price ({market_price:.4f}) is less than intrinsic value ({intrinsic:.4f})"
            )
        
        # Initialize
        sigma = initial_guess
        iterations = []
        
        for i in range(max_iterations):
            # Calculate BS price and vega
            bs_model = BlackScholesModel(self.S, self.K, self.T, self.r, sigma, self.q)
            bs_price = bs_model.option_price(option_type)
            
            greeks_calc = GreeksCalculator(self.S, self.K, self.T, self.r, sigma, self.q)
            vega = greeks_calc.vega() * 100  # Convert back to per 100% change
            
            # Calculate price difference
            price_diff = bs_price - market_price
            
            # Store iteration info
            if return_diagnostics:
                iterations.append({
                    'iteration': i,
                    'sigma': sigma,
                    'bs_price': bs_price,
                    'price_diff': price_diff,
                    'vega': vega
                })
            
            # Check convergence
            if abs(price_diff) < tolerance:
                if return_diagnostics:
                    diagnostics = {
                        'converged': True,
                        'iterations': i + 1,
                        'final_error': price_diff,
                        'iteration_history': iterations
                    }
                    return sigma, diagnostics
                return sigma
            
            # Check for numerical issues
            if vega < 1e-10:
                raise ValueError(
                    f"Vega too small ({vega:.2e}) at iteration {i}. "
                    "Cannot continue Newton-Raphson."
                )
            
            # Newton-Raphson update
            sigma_new = sigma - price_diff / vega
            
            # Ensure sigma stays positive and reasonable
            sigma_new = np.clip(sigma_new, 0.001, 5.0)
            
            # Check for oscillation
            if i > 0 and abs(sigma_new - sigma) < 1e-10:
                raise ValueError(
                    f"Sigma oscillating at iteration {i}. "
                    f"Current sigma: {sigma:.6f}, New sigma: {sigma_new:.6f}"
                )
            
            sigma = sigma_new
        
        # Did not converge
        if return_diagnostics:
            diagnostics = {
                'converged': False,
                'iterations': max_iterations,
                'final_error': price_diff,
                'iteration_history': iterations
            }
            return sigma, diagnostics
        
        raise ValueError(
            f"Failed to converge after {max_iterations} iterations. "
            f"Final error: {price_diff:.6f}, Final sigma: {sigma:.6f}"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ImpliedVolatilitySolver(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, q={self.q})"
        )
