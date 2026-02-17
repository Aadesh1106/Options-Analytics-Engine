"""
Greeks Calculator

Calculates option Greeks (sensitivities) using analytical formulas.
All Greeks are calculated using the Black-Scholes framework.
"""

import numpy as np
from typing import Union, Dict
from ..utils.helpers import (
    standard_normal_cdf,
    standard_normal_pdf,
    validate_option_params
)


class GreeksCalculator:
    """
    Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho).
    
    Greeks measure the sensitivity of option prices to various parameters:
    - Delta: Sensitivity to underlying price
    - Gamma: Rate of change of Delta
    - Vega: Sensitivity to volatility
    - Theta: Time decay
    - Rho: Sensitivity to interest rate
    
    Attributes:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate
        sigma: Volatility
        q: Dividend yield (default 0)
    """
    
    def __init__(
        self,
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: float,
        sigma: Union[float, np.ndarray],
        q: float = 0.0
    ):
        """
        Initialize Greeks calculator.
        
        Args:
            S: Current stock price(s)
            K: Strike price(s)
            T: Time to maturity in years
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
        """
        self.S = np.asarray(S)
        self.K = np.asarray(K)
        self.T = np.asarray(T)
        self.r = r
        self.sigma = np.asarray(sigma)
        self.q = q
        
        # Validate scalar inputs
        if self.S.ndim == 0 and self.K.ndim == 0 and self.T.ndim == 0 and self.sigma.ndim == 0:
            validate_option_params(
                self.S.item(),
                self.K.item(),
                self.T.item(),
                self.r,
                self.sigma.item()
            )
    
    def _calculate_d1_d2(self) -> tuple:
        """Calculate d1 and d2 parameters."""
        sqrt_T = np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2
    
    def delta(self, option_type: str) -> Union[float, np.ndarray]:
        """
        Calculate Delta: ∂V/∂S
        
        Call Delta: e^(-qT) * N(d₁)
        Put Delta: -e^(-qT) * N(-d₁)
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Delta value(s)
        """
        d1, _ = self._calculate_d1_d2()
        discount = np.exp(-self.q * self.T)
        
        option_type = option_type.lower()
        if option_type == 'call':
            delta_val = discount * standard_normal_cdf(d1)
        elif option_type == 'put':
            delta_val = -discount * standard_normal_cdf(-d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        return float(delta_val) if delta_val.ndim == 0 else delta_val
    
    def gamma(self) -> Union[float, np.ndarray]:
        """
        Calculate Gamma: ∂²V/∂S²
        
        Gamma = e^(-qT) * φ(d₁) / (S * σ * √T)
        
        Same for calls and puts.
        
        Returns:
            Gamma value(s)
        """
        d1, _ = self._calculate_d1_d2()
        sqrt_T = np.sqrt(self.T)
        discount = np.exp(-self.q * self.T)
        
        gamma_val = (discount * standard_normal_pdf(d1)) / (self.S * self.sigma * sqrt_T)
        
        return float(gamma_val) if gamma_val.ndim == 0 else gamma_val
    
    def vega(self) -> Union[float, np.ndarray]:
        """
        Calculate Vega: ∂V/∂σ
        
        Vega = S * e^(-qT) * φ(d₁) * √T
        
        Same for calls and puts.
        Note: Vega is typically expressed per 1% change in volatility.
        
        Returns:
            Vega value(s) (per 1% volatility change)
        """
        d1, _ = self._calculate_d1_d2()
        sqrt_T = np.sqrt(self.T)
        discount = np.exp(-self.q * self.T)
        
        # Vega per 1% change in volatility
        vega_val = self.S * discount * standard_normal_pdf(d1) * sqrt_T / 100
        
        return float(vega_val) if vega_val.ndim == 0 else vega_val
    
    def theta(self, option_type: str) -> Union[float, np.ndarray]:
        """
        Calculate Theta: ∂V/∂t
        
        Measures time decay (typically negative for long options).
        Note: Theta is typically expressed per day.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Theta value(s) (per day)
        """
        d1, d2 = self._calculate_d1_d2()
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)
        discount_r = np.exp(-self.r * self.T)
        
        option_type = option_type.lower()
        
        # Common term
        term1 = -(self.S * discount_q * standard_normal_pdf(d1) * self.sigma) / (2 * sqrt_T)
        
        if option_type == 'call':
            term2 = self.q * self.S * discount_q * standard_normal_cdf(d1)
            term3 = -self.r * self.K * discount_r * standard_normal_cdf(d2)
            theta_val = term1 - term2 + term3
        elif option_type == 'put':
            term2 = self.q * self.S * discount_q * standard_normal_cdf(-d1)
            term3 = self.r * self.K * discount_r * standard_normal_cdf(-d2)
            theta_val = term1 + term2 - term3
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        # Convert to per-day theta (divide by 365)
        theta_val = theta_val / 365
        
        return float(theta_val) if theta_val.ndim == 0 else theta_val
    
    def rho(self, option_type: str) -> Union[float, np.ndarray]:
        """
        Calculate Rho: ∂V/∂r
        
        Sensitivity to interest rate changes.
        Note: Rho is typically expressed per 1% change in interest rate.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Rho value(s) (per 1% rate change)
        """
        _, d2 = self._calculate_d1_d2()
        discount_r = np.exp(-self.r * self.T)
        
        option_type = option_type.lower()
        if option_type == 'call':
            rho_val = self.K * self.T * discount_r * standard_normal_cdf(d2)
        elif option_type == 'put':
            rho_val = -self.K * self.T * discount_r * standard_normal_cdf(-d2)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        # Rho per 1% change in interest rate
        rho_val = rho_val / 100
        
        return float(rho_val) if rho_val.ndim == 0 else rho_val
    
    def all_greeks(self, option_type: str) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate all Greeks at once.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with all Greeks
        """
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GreeksCalculator(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, q={self.q})"
        )
