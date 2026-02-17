"""
Utility functions and helpers for numerical stability and common operations.
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Tuple


def standard_normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Cumulative distribution function of standard normal distribution.
    
    Args:
        x: Input value(s)
        
    Returns:
        CDF value(s)
    """
    return norm.cdf(x)


def standard_normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Probability density function of standard normal distribution.
    
    Args:
        x: Input value(s)
        
    Returns:
        PDF value(s)
    """
    return norm.pdf(x)


def validate_option_params(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> None:
    """
    Validate option pricing parameters.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if S <= 0:
        raise ValueError(f"Stock price must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time to maturity must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility must be positive, got {sigma}")
    if sigma > 5.0:
        raise ValueError(f"Volatility seems unreasonably high: {sigma}")


def calculate_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> Tuple[float, float]:
    """
    Calculate d1 and d2 for Black-Scholes formula.
    
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        Tuple of (d1, d2)
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from periodic returns.
    
    Args:
        returns: Array of periodic returns
        periods_per_year: Number of periods in a year (252 for daily)
        
    Returns:
        Annualized return
    """
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from periodic returns.
    
    Args:
        returns: Array of periodic returns
        periods_per_year: Number of periods in a year (252 for daily)
        
    Returns:
        Annualized volatility
    """
    return np.std(returns) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        Maximum drawdown (positive value)
    """
    cumulative_wealth = 1 + cumulative_returns
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    return abs(np.min(drawdown))


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for data.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))
    z_score = norm.ppf((1 + confidence) / 2)
    margin = z_score * std_error
    return mean - margin, mean + margin


def ensure_positive(value: float, epsilon: float = 1e-10) -> float:
    """
    Ensure a value is positive for numerical stability.
    
    Args:
        value: Input value
        epsilon: Minimum positive value
        
    Returns:
        Max of value and epsilon
    """
    return max(value, epsilon)
