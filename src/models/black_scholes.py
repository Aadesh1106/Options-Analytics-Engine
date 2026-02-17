"""
Black-Scholes Option Pricing Model

Implements the Black-Scholes-Merton formula for European options with
vectorized operations for efficient batch pricing.
"""

import numpy as np
from typing import Union
from ..utils.helpers import (
    standard_normal_cdf,
    validate_option_params,
    calculate_d1_d2
)


class BlackScholesModel:
    """
    Black-Scholes option pricing model for European options.
    
    The model assumes:
    - Log-normal distribution of stock prices
    - Constant volatility and risk-free rate
    - No dividends
    - Frictionless markets
    - Continuous trading
    
    Attributes:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        q: Dividend yield (annual, default 0)
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
        Initialize Black-Scholes model.
        
        Args:
            S: Current stock price(s)
            K: Strike price(s)
            T: Time to maturity in years
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
            q: Dividend yield (annual, default 0)
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
        """
        Calculate d1 and d2 parameters.
        
        Returns:
            Tuple of (d1, d2)
        """
        sqrt_T = np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2
    
    def call_price(self) -> Union[float, np.ndarray]:
        """
        Calculate European call option price using Black-Scholes formula.
        
        C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)
        
        Returns:
            Call option price(s)
        """
        d1, d2 = self._calculate_d1_d2()
        
        call = (
            self.S * np.exp(-self.q * self.T) * standard_normal_cdf(d1) -
            self.K * np.exp(-self.r * self.T) * standard_normal_cdf(d2)
        )
        
        return float(call) if call.ndim == 0 else call
    
    def put_price(self) -> Union[float, np.ndarray]:
        """
        Calculate European put option price using Black-Scholes formula.
        
        P = Ke^(-rT)N(-d₂) - S₀e^(-qT)N(-d₁)
        
        Returns:
            Put option price(s)
        """
        d1, d2 = self._calculate_d1_d2()
        
        put = (
            self.K * np.exp(-self.r * self.T) * standard_normal_cdf(-d2) -
            self.S * np.exp(-self.q * self.T) * standard_normal_cdf(-d1)
        )
        
        return float(put) if put.ndim == 0 else put
    
    def option_price(self, option_type: str) -> Union[float, np.ndarray]:
        """
        Calculate option price for specified type.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Option price(s)
            
        Raises:
            ValueError: If option_type is not 'call' or 'put'
        """
        option_type = option_type.lower()
        if option_type == 'call':
            return self.call_price()
        elif option_type == 'put':
            return self.put_price()
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    
    def intrinsic_value(self, option_type: str) -> Union[float, np.ndarray]:
        """
        Calculate intrinsic value of option.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Intrinsic value(s)
        """
        option_type = option_type.lower()
        if option_type == 'call':
            value = np.maximum(self.S - self.K, 0)
        elif option_type == 'put':
            value = np.maximum(self.K - self.S, 0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        return float(value) if value.ndim == 0 else value
    
    def time_value(self, option_type: str) -> Union[float, np.ndarray]:
        """
        Calculate time value of option.
        
        Time Value = Option Price - Intrinsic Value
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Time value(s)
        """
        return self.option_price(option_type) - self.intrinsic_value(option_type)
    
    def moneyness(self) -> Union[float, np.ndarray]:
        """
        Calculate moneyness (S/K).
        
        Returns:
            Moneyness ratio(s)
        """
        m = self.S / self.K
        return float(m) if m.ndim == 0 else m
    
    def is_itm(self, option_type: str) -> Union[bool, np.ndarray]:
        """
        Check if option is in-the-money.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Boolean or array of booleans
        """
        option_type = option_type.lower()
        if option_type == 'call':
            result = self.S > self.K
        elif option_type == 'put':
            result = self.S < self.K
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        
        return bool(result) if result.ndim == 0 else result
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"BlackScholesModel(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, q={self.q})"
        )
